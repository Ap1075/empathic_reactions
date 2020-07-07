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
import json
import itertools

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
file = '20200306_1530_919599800518_12220083.mp3.xyz'
with open('/media/armaan/AP-HD1/Battlefield/gnani/datasets/Transcripts/20200306_1530_919599800518_12220083.mp3.xyz') as f: 
    d = json.load(f)

my_test = json_normalize(d['results'],record_path ='alternatives')
my_test.rename(columns={"transcript":"essay"}, inplace=True)
col_types = {'startTime':float, 'endTime':float, 'essay':object, 'confidence':int, 'speaker':int}
my_test = my_test.astype(col_types) 
# my_test = pd.read_csv('/media/armaan/AP-HD1/Battlefield/gnani/datasets/Transcripts/test_emots.csv')
my_test = my_test[my_test['speaker']==2] ## specifying only agent's part
########################################

# print(data, data.shape)
# assert(False)

results_df=pd.DataFrame(
	index=['cnn', 'ffn', 'ridge', 'maxent'], 
	columns=['empathy', 'empathy_bin', 'distress', 'distress_bin'])


embs=common.get_facebook_fasttext_common_crawl(vocab_limit=None)

TARGETS=['empathy', 'distress']






# features_train_centroid=fe.embedding_centroid(train.essay, embs)
# features_train_matrix=fe.embedding_matrix(train.essay, embs, common.TIMESTEPS)

# features_test_centroid=fe.embedding_centroid(test.essay, embs)
# features_test_matrix=fe.embedding_matrix(test.essay, embs, common.TIMESTEPS)

FEATURES_MATRIX=fe.embedding_matrix(my_test.essay, embs, common.TIMESTEPS)
FEATURES_CENTROID=fe.embedding_centroid(my_test.essay, embs)

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
							num_filters=100, 
							learning_rate=1e-3,
							dropout_conv=.5, 
							problem='regression'),

	'ffn': lambda:	common.get_ffn(
							units=[300,256, 128,1], 
							dropout_hidden=.5,
							dropout_embedding=.2, 
							learning_rate=1e-3,
							problem='regression'),

	'ridge': lambda: RidgeCV(
							alphas=[1, 5e-1, 1e-1,5e-2, 1e-2, 5e-3, 1e-3,5e-4, 1e-4])
}



early_stopping=keras.callbacks.EarlyStopping(monitor='val_loss',
                              min_delta=0,
                              patience=20,
                              verbose=0, mode='auto')


num_splits=10

performancens={name:pd.DataFrame(columns=['empathy', 'distress'], 
	index=range(1,num_splits+1)) for name in MODELS.keys()}


emp_dis_df = pd.DataFrame(columns=['file_name','empathy', 'distress'])
emp_dis_df['file_name'] = file

kf_iterator=KFold(n_splits=num_splits, shuffle=True, random_state=42)
# for i, splits in enumerate(kf_iterator.split(data)):
	# train,test=splits

# test = pd.read_csv('/media/armaan/AP-HD1/Battlefield/gnani/datasets/Transcripts/test_emots.csv')

k.clear_session()

for target in TARGETS:
	# print(target)

	# labels_train=data[target][train]
	# labels_test=data[target][test]

	# features_train_centroid=FEATURES_CENTROID[train]
	# features_train_matrix=FEATURES_MATRIX[train]

	features_test_centroid=FEATURES_CENTROID
	features_test_matrix=FEATURES_MATRIX


	# print(labels_train)
	# print(features_train_matrix)

	for model_name, model_fun in MODELS.items():
		# print(model_name)
		model=model_fun()


		#	TRAINING
		# if model_name=='cnn':
		# 	model.fit(	features_train_matrix, 
		# 				labels_train,
		# 				epochs=200, 
		# 				batch_size=32, 
		# 				validation_split=.1, 
		# 				callbacks=[early_stopping])

		# elif model_name=='ffn':
		# 	model.fit(	features_train_centroid,
		# 				labels_train,
		# 				epochs=200, 
		# 				validation_split=.1, 
		# 				batch_size=32, 
		# 				callbacks=[early_stopping])

		# elif model_name=='ridge':
		# 	model.fit(	features_train_centroid,
		# 				labels_train)

		# else:
		# 	raise ValueError('Unkown model name encountered.')

		#	PREDICTION
		if model_name=='cnn':
			model.load_weights('./model_cnn.h5')
			pred=model.predict(features_test_matrix)
		else:
			continue
			# pred=model.predict(features_test_centroid)

		#	SCORING
		# result=correlation(true=labels_test, pred=pred)

		#	RECORD
		# row=model_name
		# column=LABELS[target][problem]
		# results_df.loc[row,column]=result
		# print(results_df)
		# performancens[model_name].loc[i+1,target]=result
		# print(performancens[model_name])
		print("#####PREDICTIONS for {0} {1} added to DF".format(model_name,target))
		emp_dis_df[target] = list(itertools.chain.from_iterable(pred))
		# print(pred)


emp_dis_df.to_csv('fina_results.csv', index=False)
#average results data frame
if not os.path.isdir('results'):
	os.makedirs('results')
for key, performance in performancens.items():
	performance['mean']=performance.mean(axis=1)
	mean=performance.mean(axis=0)
	stdev=performance.std(axis=0)
	performance.loc['stdev']=stdev
	performance.loc['mean']=mean
	performance.to_csv('results/{}_edited.tsv'.format(key), sep='\t')

# results_df.to_csv('results.tsv', sep='\t')





