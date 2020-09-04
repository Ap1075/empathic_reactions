import pandas as pd
import os
import util
import modeling.feature_extraction as fe
from modeling import common
from scipy import stats as st
from sklearn import metrics
import numpy as np
import keras
from sklearn.linear_model import RidgeCV, LogisticRegressionCV
from sklearn.model_selection import KFold
from pandas import json_normalize
import json, csv
import itertools
import pickle
## extra imports to set GPU options
import tensorflow as tf
from keras import backend as k
 
###################################
# TensorFlow wizardry
config = tf.ConfigProto()
 
# Don't pre-allocate memory; allocate as-needed
config.gpu_options.allow_growth = True
 
# Only allow a total of half the GPU memory to be allocated
config.gpu_options.per_process_gpu_memory_fraction = 0.5
 
# Create a session with the above options specified.
k.tensorflow_backend.set_session(tf.Session(config=config))
###################################


#######		Setting up data		########
# train, dev, test=util.train_dev_test_split(util.get_messages())
# data=pd.concat([train,test], axis=0) #excluding dev set from CV
# data=data.reset_index(drop=True)
########################################

# print(data, data.shape)
# assert(False)

results_df=pd.DataFrame(
	index=['cnn', 'ffn', 'ridge', 'maxent'], 
	columns=['empathy', 'empathy_bin', 'distress', 'distress_bin'])


embs=common.get_facebook_fasttext_common_crawl(vocab_limit=None)

TARGETS=['empathy', 'distress']
# TARGETS=['empathy', 'distress'] # for classification





# features_train_centroid=fe.embedding_centroid(train.essay, embs)
# features_train_matrix=fe.embedding_matrix(train.essay, embs, common.TIMESTEPS)

# features_test_centroid=fe.embedding_centroid(test.essay, embs)
# features_test_matrix=fe.embedding_matrix(test.essay, embs, common.TIMESTEPS)

# LABELS={
# 	'empathy':{'classification':'empathy_bin', 'regression':'empathy'},
# 	'distress':{'classification':'distress_bin', 'regression':'distress'}
# }

def f1_score(true, pred):
	pred=np.where(pred.flatten() >.5 ,1,0)
	result=metrics.precision_recall_fscore_support(
		y_true=true, y_pred=pred, average='micro')
	return result[2]

def correlation(true, pred):
	pred=pred.flatten()
	result=st.pearsonr(true,pred)
	return result[0]

# SCORE={
# 	'classification': f1_score,
# 	'regression':correlation
# }


MODELS={
	'cnn':lambda:common.get_cnn(
							input_shape=[common.TIMESTEPS,300], 
							num_outputs=1, 
							num_filters=256, 
							learning_rate=1e-3,
							dropout_conv=.5, 
							problem='regression'),

	# 'ffn': lambda:	common.get_ffn(
	# 						units=[300,256, 128,1], 
	# 						dropout_hidden=.5,
	# 						dropout_embedding=.2, 
	# 						learning_rate=1e-3,
	# 						problem='regression'),

	# 'ridge': lambda: RidgeCV(
	# 						alphas=[1, 5e-1, 1e-1,5e-2, 1e-2, 5e-3, 1e-3,5e-4, 1e-4])
}



early_stopping=keras.callbacks.EarlyStopping(monitor='val_loss',
							  min_delta=0,
							  patience=20,
							  verbose=0, mode='auto')

num_splits = 10
performancens={name:pd.DataFrame(columns=['empathy', 'distress'], 
	index=range(1,num_splits+1)) for name in MODELS.keys()}


# emp_dis_df = pd.DataFrame(columns=['file_name','empathy', 'distress','total_dead_air'])
out_csv = 'softSkills_July.csv'
if not os.path.isfile(out_csv):
	with open (out_csv, 'a') as csvfile:
		headers = ['file_name','call_duration','total_dead_air','interrupt_count','rate_of_speech','empathy', 'distress', 'interrupt_rate', 'ratio_dead_air']
		wr = csv.writer(csvfile, dialect='excel')
		wr.writerow(headers)


k.clear_session()
# path_to_json = '/media/armaan/AP-HD2/Battlefield/gnani/tvs_trans/CSPLITBPORecording7thJuly'
# path_to_json = '/media/armaan/AP-HD2/Battlefield/gnani/audio'
path_to_json = '/media/armaan/AP-HD2/Battlefield/gnani/datasets/all_transcripts'
json_files = [pos_json for pos_json in os.listdir(path_to_json) if pos_json.endswith('.json')]
#################################
print('this is models cnn',MODELS['cnn']().load_weights('./models/hindi2vec_full_empathy.hdf5'))
model_emp =  MODELS['cnn']()
model_dis = MODELS['cnn']()
model_emp.load_weights('./models/hindi2vec_full_empathy.hdf5')
model_dis.load_weights('./models/hindi2vec_full_distress.hdf5')

#################################
with open('/media/armaan/AP-HD2/Battlefield/gnani/chatbot_ner/data/switched.txt', 'rb') as f:
    switched_list = pickle.load(f)

for n,file in enumerate(json_files):
	if file in switched_list:
		customer_id=2
		agent_id=1
	else:
		customer_id = 1
		agent_id = 2
	if n%100==0:
		print("Processed {} files".format(n))
	full_filename = "%s/%s" % (path_to_json, file)

	with open(full_filename,'r') as fi:
		d = json.load(fi)

	my_test = json_normalize(d['results'],record_path ='alternatives')
	if 'transcript' not in my_test.columns:
		continue
	my_test.rename(columns={"transcript":"essay"}, inplace=True)
	col_types = {'startTime':float, 'endTime':float, 'essay':object, 'confidence':float, 'speaker':int}
	my_test = my_test.astype(col_types) 
	if len(my_test[my_test['speaker']==agent_id].essay) <= 1:
		continue
	# my_test = pd.read_csv('/media/armaan/AP-HD1/Battlefield/gnani/datasets/Transcripts/test_emots.csv')

	total_da = 0
	interrupt_count = 0
	rate_of_speech = (len(my_test[my_test['speaker']==agent_id].essay.str.cat(sep=' ').split())*60)/((my_test[my_test['speaker']==agent_id]['endTime']-my_test[my_test['speaker']==agent_id]['startTime']).sum())
	# emp_dis_df = pd.DataFrame(columns=['file_name','call_duration','total_dead_air','empathy', 'distress'])
	call_duration = my_test.endTime.iloc[-1] - my_test.startTime.iloc[0]
	for i in range(len(my_test)-1): 
		total_da +=( my_test.startTime.iloc[i+1]- my_test.endTime.iloc[i])
		if (my_test.startTime.iloc[i+1] < my_test.endTime.iloc[i]) and (my_test.speaker.iloc[i+1]==agent_id) and (my_test.speaker.iloc[i]==customer_id):
			interrupt_count +=1

	# if file in switched_list:
	# 	my_test = my_test[my_test['speaker']==1]
	# else:
	my_test = my_test[my_test['speaker']==agent_id] ## specifying only agent's part

	FEATURES_MATRIX=fe.embedding_matrix(pd.Series(my_test.essay.str.cat(sep=' ')), embs, common.TIMESTEPS)
	FEATURES_CENTROID=fe.embedding_centroid(pd.Series(my_test.essay.str.cat(sep=' ')), embs)

	features_test_centroid=FEATURES_CENTROID
	features_test_matrix=FEATURES_MATRIX

	pred_emp=model_emp.predict(features_test_matrix)
	pred_dis=model_dis.predict(features_test_matrix)

	emp_score  = round(pred_emp[0][0], 4)
	dis_score = round(pred_dis[0][0], 4)

	with open (out_csv, 'a') as csvfile:
		wr = csv.writer(csvfile, dialect='excel')

		wr.writerow([path_to_json.split('/')[-1]+'/'+file, round(call_duration,4), round(total_da,4), interrupt_count, round(rate_of_speech,4), emp_score, dis_score,round(interrupt_count*60/call_duration,4),round(total_da/call_duration,4) ])
