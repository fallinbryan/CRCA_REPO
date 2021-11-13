
import pandas as pd   
from scipy.spatial.distance import cosine as cosine_distance
import numpy as np
import datetime as dt
import pickle 

import pickle 

keyed_embeddings = None 
transaction_embeddings = None

with open('data/keyed_alarm_embeddings.pickle','rb') as f:
  keyed_embeddings = pickle.load(f)

with open('data/transaction_embeddings.pickle', 'rb') as f:
  transaction_embeddings = pickle.load(f)

def ClassifyAlarm(sentVec, embeddings):
  classification = None
  min = 9999
  for k in embeddings:
    d = cosine_distance(sentVec, embeddings[k])
    if d < min:
      min = d
      classification = k
  return classification, d
  
  
  
class RCA:
  def __init__(self, distinct_alarms, transactions):
    self.transactions = {}
    for tid in transactions:
      self.transactions[tid] = transactions[tid]['ALR']
    self.transaction_count = float(len(transactions))

    self.alr_transction_probabilities = {}
    for alr in distinct_alarms:
      count = 0
      for tid in transactions:
        if alr in self.transactions[tid]:
          count += 1
      self.alr_transction_probabilities[alr] = count / self.transaction_count
      
  def _PofList(self, arl_list):
    d_list = list(set(arl_list))
    list_len = len(d_list)
    count = 0
    t = []
    for tid in self.transactions:
      items_in_tid = 0
      for item in d_list:
        if item in self.transactions[tid]:
          items_in_tid += 1
      if items_in_tid == list_len:
        count +=1
      t.append(items_in_tid)
    if count == 0:
      return np.mean(t) / self.transaction_count
    return count / self.transaction_count

  def ScoreTransaction(self, transaction):
    Parr = [self.alr_transction_probabilities[t] for t in transaction['ALR']]
    rca = {'EventId': transaction['EventId'], 'ALR': transaction['ALR']}
    Pcond =[]
    pAll = self._PofList(transaction['ALR'])
    for i, pA in enumerate(Parr):
      tCopy = [x for x in transaction['ALR']]
      tCopy.pop(i)
      pB = self._PofList(tCopy)
      if pB==0:
        Pcond.append(1.0e-26)
      else:
        Pcond.append(  pA * pAll/pB )

    rca['RCA_SCORE'] = Pcond
    return rca

  def ScoreCluster(self, alarm_list, alr_idx):
    if len(alarm_list) <= 1:
      return [1]
    Parr = []
    alrms = [t[alr_idx] for t in alarm_list]
    for alr in alrms:
      if alr not in self.alr_transction_probabilities:
        Parr.append(1.0e-26)
      else: 
        Parr.append(self.alr_transction_probabilities[alr])
    #print(Parr)
    Pcond =[]
    pAll = self._PofList( [t[alr_idx] for t in alarm_list] )
    for i, pA in enumerate(Parr):
      tCopy = [t[alr_idx] for t in alarm_list]
      #print(tCopy)
      tCopy.pop(i)
      pB = self._PofList(tCopy)
      if pB==0:
        Pcond.append(1.0e-26)
      else:
        Pcond.append(  pA * pAll/pB )
    return Pcond
 
def GetRca(clusterList, alr_idx):
  return RcaAlgo.ScoreCluster(clusterList, alr_idx)

 
    
alr_df = pd.read_csv('data/final_clustering2.csv')
test_df = pd.read_csv('data/AlarmTestSet.csv')

test_df['EventDt'] = pd.to_datetime(test_df['EventDt'])
test_df['ClearDt'] = pd.to_datetime(test_df['ClearDt'])
test_df['EVENT_DT_epoch'] = (test_df['EventDt'] - dt.datetime(1970,1,1)).dt.total_seconds()
test_df['EVENT_CLR_epoch'] = (test_df['ClearDt']- dt.datetime(1970,1,1)).dt.total_seconds()

test_df.sort_values('EVENT_DT_epoch', inplace=True)

print('data loaded')


transction_ids = set(alr_df['TransactionClusterId'].values) 


number_of_transactions = len(transction_ids)

distinct_alarms = set(alr_df['sentence'].values)


transaction_alarms = {}
for tid in transction_ids:
  transaction_alarms[tid] = {'EventId':[], 'ALR': []}
for eid, tid, alr in alr_df[['EventId','TransactionClusterId','sentence']].values:
  transaction_alarms[tid]['EventId'].append(eid)
  transaction_alarms[tid]['ALR'].append(alr) 

print('transactions compiled')

RcaAlgo = RCA(distinct_alarms, transaction_alarms)


print('rca algorithm initialized. running on-line test')



current_vec = []
current_cluster = []
prev_time = 0
prev_dist = 1
current_correlationId = 0
epsilon = 1.05
min_time = 7.0
probability_threshold = 0.0005
clustered_df_dic = {'EventId': [], "ClusterId": [], "RCA_Score": [], "Class": [], "DistanceToClass":[], "AlarmsInCluster":[], "MaxRCAInCluster":[], "DropReason":[]}

counter = 0
slice_at= 200#len(test_df)-1

for  eventId, eventDt, ClearDt, sentence, subj in test_df[["EventId","EVENT_DT_epoch","EVENT_CLR_epoch","sentence", 'SUBJECT']].values[0:slice_at]:
  
  k=0 
  d=-1
  if counter == 0:
    prev_time = eventDt
  t_delta = eventDt - prev_time
  prev_time = eventDt
  
  if counter % 200 == 0:
    print(f'{100 * counter/slice_at}% Complete')
    print(f'Saving data to file')
    test_result_df =  pd.DataFrame(clustered_df_dic)
    test_df.join(test_result_df.set_index('EventId'), on='EventId', how='inner').set_index('EventId').to_csv('data/test_result4.csv')

  counter += 1

  if sentence in keyed_embeddings:
    current_vec.append(keyed_embeddings[sentence])
  
  if len(current_vec) > 0:
    new_vec = np.array(current_vec).mean(axis=0)

    k, d = ClassifyAlarm(new_vec, transaction_embeddings)


  if d >  epsilon or t_delta > min_time:
    drop_reason = ""
    
    scores = GetRca(current_cluster, 1)
    max_score = max(scores)
    this_corrleationId = current_correlationId

    #alarms_in_cluster = (np.array(scores) > probability_threshold).sum()

    if len(current_cluster) == 0:
      drop_reason = "culster len == 0"
      clustered_df_dic['EventId'].append(eventId)
      clustered_df_dic['ClusterId'].append(-1)
      clustered_df_dic['RCA_Score'].append(1)
      clustered_df_dic['Class'].append(k)
      clustered_df_dic['DistanceToClass'].append(d)
      clustered_df_dic['AlarmsInCluster'].append(0)
      clustered_df_dic['MaxRCAInCluster'].append(1)
      clustered_df_dic['DropReason'].append(drop_reason)
      continue
    if len(current_cluster) == 1:
      drop_reason = "culster len == 1"
      this_corrleationId = -1
      alarms_in_cluster = 0

    for i, score in enumerate(scores):
      this_corrleationId = current_correlationId
      alarms_in_cluster = len(current_cluster)
      drop_reason = ""
      clustered_df_dic['EventId'].append(current_cluster[i][0])
      if score < probability_threshold:
        this_corrleationId = -1
        alarms_in_cluster = 0
        drop_reason =f'rca score: {score} < threshold: {probability_threshold}'
      clustered_df_dic['ClusterId'].append(this_corrleationId)
      clustered_df_dic['RCA_Score'].append(score)
      clustered_df_dic['Class'].append(current_cluster[i][4])
      clustered_df_dic['DistanceToClass'].append(current_cluster[i][5])
      clustered_df_dic['AlarmsInCluster'].append(alarms_in_cluster)
      clustered_df_dic['MaxRCAInCluster'].append(max_score)
      clustered_df_dic['DropReason'].append(drop_reason)

    current_correlationId += 1
    current_cluster = []
    current_timedeltas = []
    current_vec = []
  
  current_cluster.append([eventId, sentence, subj, current_correlationId, k, d])
  prev_dist = d
  #print(sentence, k, d, t_delta, current_correlationId)
  #print(eventId, eventDt, ClearDt, Severity, subject)


print('run complete, writing result file')


test_result_df =  pd.DataFrame(clustered_df_dic)
test_df.join(test_result_df.set_index('EventId'), on='EventId', how='inner').set_index('EventId').to_csv('data/test_result4.csv')



print('done')
