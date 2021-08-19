random_seed = 1234

from IPython.display import Audio, display

import json
import glob
import soundfile
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import preprocess as cmi

import os
from os import path
from pydub import AudioSegment
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from scipy.signal import spectrogram
from tqdm import tqdm
from sys import argv

from catboost import CatBoostClassifier

import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
from tensorflow.keras import layers
from tensorflow.keras.models import model_from_config

from python_speech_features import fbank

def MP3toFLAC(src, dst, delete=False):
	sound = AudioSegment.from_mp3(src)
	sound.export(dst, format="flac")
	if delete:
		os.remove(src)
	return dst

classifiers = ["SVM", "CatBoost"]
print("Which classifier would you like to use? Enter the integer associated")
for i, c in enumerate(classifiers):
	print(f"{i}: {c}")
selection = int(input(""))
classifier = classifiers[selection]
window_len = 2
step = 0.1

batch_size = -1
mp3_dir = "audio_data/Bachman Sparrow 10sec Smaller"
dataset = mp3_dir.split('/')[-1]
flac_dir = "audio_data/Bachman Sparrow 10sec Smaller/FLAC"
reset_dir = True if input("Clear FLAC directory (y/n): ").lower()=="y" else False
if reset_dir:
	if path.isdir(flac_dir):
		from shutil import rmtree
		rmtree(flac_dir)
	os.mkdir(flac_dir)
	count = 0
	for file in tqdm(os.listdir(mp3_dir)):
		if batch_size >=0 and count >= batch_size:
			break		
		if file.split('.')[-1] != "mp3":
			continue
		MP3toFLAC(src=f"{mp3_dir}/{file}", dst=f"{flac_dir}/{file.split('/')[-1][:-3]+'flac'}")
		count += 1

csv_dir = "resources/bachman_labels.csv"
assert path.exists(csv_dir)

def generate_embeddings(X_train, X_test, feature_model):
	print("Making fbanks...")
	fbank_train = np.array([cmi.make_fbank(x) for x in tqdm(X_train[:,0])])
	fbank_test = np.array([cmi.make_fbank(x) for x in tqdm(X_test[:,0])])

	print("Normalizing...")
	scale =  33.15998
	X_train_normal = fbank_train[:,:40,:] / scale
	X_test_normal = fbank_test[:,:40,:] / scale

	batch_train = X_train_normal.reshape(X_train_normal.shape[0],
	                                     X_train_normal.shape[1],
	                                     X_train_normal.shape[2],
	                                     1)

	batch_test = X_test_normal.reshape(X_test_normal.shape[0],
	                                   X_test_normal.shape[1],
	                                   X_test_normal.shape[2],
	                                   1)

	run_size = 1000
	start = 0
	embeddings_train = []
	print("Embedding...")
	while start < len(batch_train):
		embeddings_train += feature_model(batch_train[start : start+run_size])[-1].numpy().tolist()
		start += run_size
	embeddings_train = tf.convert_to_tensor(embeddings_train)
	embeddings_test = feature_model(batch_test)[-1]	

	return embeddings_train, embeddings_test

from sklearn.svm import SVC
from sklearn.linear_model import Perceptron, SGDClassifier, PassiveAggressiveClassifier
from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import GridSearchCV

print("Pulling model...")
model = model_from_config(json.load(open('resources/cmi_mbam01.json', 'r')))
model.load_weights('resources/cmi_mbam01.h5')

# Let's look at the original BAM model
model.summary()

feature_layers = [layer.output for layer in model.layers[:-4]]
feature_model = tf.keras.Model(inputs=[model.input], outputs=feature_layers)

# visualize the embedding model
feature_model.summary()

if len(argv) > 1:
	if argv[-1] == "save_intermediate":
		reset_dir = True if input("Clear FLAC directory (y/n): ").lower()=="y" else False
		if reset_dir:
			if path.isdir(flac_dir):
				from shutil import rmtree
				rmtree(flac_dir)
			os.mkdir(flac_dir)
			count = 0
			for file in tqdm(os.listdir(mp3_dir)):
				if batch_size >=0 and count >= batch_size:
					break		
				if file.split('.')[-1] != "mp3":
					continue
				MP3toFLAC(src=f"{mp3_dir}/{file}", dst=f"{flac_dir}/{file.split('/')[-1][:-3]+'flac'}")
				count += 1

		csv_dir = "resources/bachman_labels.csv"
		assert path.exists(csv_dir)

		print("Processing data...")
		data, labels = cmi.preprocess_data(target=csv_dir, audio_dir=flac_dir, DEBUG=True)
		X_train, X_test, y_train, y_test = train_test_split(data, labels,
		                                                    test_size=0.20)

		# set initial values
		save_batch = 100
		start = 0

		# create and open files
		intermediate_dir = f"intermediate_data/{dataset}_{window_len}_{step}_{batch_size}"
		if not path.isdir(intermediate_dir):
			os.mkdir(intermediate_dir)
		encode = 'a'
		if not path.exists(f"{intermediate_dir}/embeddings.txt"):
			encode = 'w'
		embed_file = open(f"{intermediate_dir}/embeddings.txt", encode)
		encode = 'a'
		if not os.path.exists(f"{intermediate_dir}/labels.txt"):
			encode = 'w'
		label_file = open(f"{intermediate_dir}/labels.txt", encode)

		# create batches
		full_X = np.array(X_train.tolist() + X_test.tolist())
		full_y = np.array(y_train.tolist() + y_test.tolist())
		while start < len(full_X):
			print(f"Running {start} / {len(full_X)}")
			X_batch = full_X[start : start+save_batch]
			y_batch = full_y[start : start+save_batch]
			X_batch_train, X_batch_test, y_batch_train, y_batch_test = train_test_split(X_batch, y_batch, test_size=0.2)
			# generate embeddings
			embeddings_train, embeddings_test = generate_embeddings(X_batch_train, X_batch_test, feature_model)
			# write results
			for e in embeddings_train.numpy().tolist() + embeddings_test.numpy().tolist():
				e_string = ','.join([str(val) for val in e])
				embed_file.write(f"{e_string}\n")
			for l in y_batch_train.tolist() + y_batch_test.tolist():
				l_string = str(l)
				label_file.write(f"{l_string}\n")

			# move start point
			start += save_batch
		
		# close files
		embed_file.close()
		label_file.close()

	elif argv[-1] == "load_intermediate":
		intermediate_dir = f"intermediate_data/{dataset}_{window_len}_{step}_{batch_size}"
		embed_file = open(f"{intermediate_dir}/embeddings.txt")
		label_file = open(f"{intermediate_dir}/labels.txt")
		batch_size = 1000
		predictor = None
		if classifier == "SVM":
			param_grid = {'C': [0.1, 1, 10, 1e2, 1e3],
			              'gamma': [10, 100, 1000], }
			clf = SVC(kernel="rbf", class_weight="balanced", verbose=True)			

		elif classifier == "CatBoost":
			param_grid = {
				"depth": [6,8,10],
				"learning_rate": [0.01, 0.05, 0.1],
				"iterations": [30, 50, 100]
			}
			clf = CatBoostClassifier(verbose=True)

		predictor = GridSearchCV(clf, param_grid, cv=2, verbose=50)
		X = np.array([[float(e) for e in line.split(',')] for line in embed_file.read().strip().split('\n')])
		l = np.array([int(l) for l in label_file.read().strip().split('\n')])
		X_train, _, y_train, _ = train_test_split(X, l, test_size=0.2)
		print("Training...")
		predictor.fit(X_train, y_train)
		print("Trained!")
		# else:
		# 	if classifier == "Perceptron":
		# 		clf = Perceptron(verbose=50)
		# 		param_grid = {"alpha": [0.0001, 0.001, 0.01, 0.1, 1],
		# 					  "eta0": [0.5, 1, 2, 4, 8]}
		# 	elif classifier == "SGD":
		# 		clf = SGDClassifier(verbose=50)
		# 		param_grid = {
		# 			"loss": ["hinge", "log", "squared_hinge"],
		# 			"alpha": [0.0001, 0.001, 0.01, 0.1, 1]
		# 		}
		# 	elif classifier == "PassiveAggressiveClassifier":
		# 		clf = PassiveAggressiveClassifier(verbose=50)
		# 		param_grid = {
		# 			"C": [0.01, 0.1, 1, 10, 100],
		# 			"fit_intercept": [True, False]
		# 		}

			# label_file.close()
			# label_file = open(f"{intermediate_dir}/labels.txt")
			# data_count = len(label_file.read().split('\n'))
			# for i in range(0, data_count, batch_size):
				# embed_file.close()
				# embed_file = open(f"{intermediate_dir}/embeddings.txt")
				# label_file.close()
				# label_file = open(f"{intermediate_dir}/labels.txt")
				# data_count = len(label_file.read().split('\n'))
				# print(f"Fitting {classifier} on {i} / {data_count}")
				# X = np.array([[float(e) for e in line.split(',')] for line in embed_file.read().strip().split('\n')[i : i+batch_size]])
				# label_file.close()
				# label_file = open(f"{intermediate_dir}/labels.txt")
				# l = np.array([int(l) for l in label_file.read().strip().split('\n')[i : i+batch_size]])
				# X_batch_train, _, y_batch_train, _ = train_test_split(X, l, test_size=0.2)
				# predictor.partial_fit(X_batch_train, y_batch_train, [0,1])
			
			# predictor = GridSearchCV(clf, param_grid, cv=2, verbose=50)
			
			# X = np.array([[float(e) for e in line.split(',')] for line in open(f"{intermediate_dir}/embeddings.txt").read().strip().split('\n')])
			# l = np.array([int(l) for l in label_file.read().strip().split('\n')])
			# X_train, _, y_train, _ = train_test_split(X, l, test_size=0.2)
			# predictor.fit(X_train, y_train)

		embed_file.close()
		embed_file = open(f"{intermediate_dir}/embeddings.txt")
		X = np.array([[float(e) for e in line.split(',')] for line in embed_file.read().strip().split('\n')])
		label_file.close()
		label_file = open(f"{intermediate_dir}/labels.txt")
		l = np.array([int(l) for l in label_file.read().strip().split('\n')])
		y_pred = predictor.predict(X)

		print(classification_report(l, y_pred))

		from sklearn.metrics import plot_confusion_matrix
		import matplotlib.pyplot as plt
		plot_confusion_matrix(predictor, X, l)
		plt.show()

		embed_file.close()
		label_file.close()

else:
	reset_dir = True if input("Clear FLAC directory (y/n): ").lower()=="y" else False
	if reset_dir:
		if path.isdir(flac_dir):
			from shutil import rmtree
			rmtree(flac_dir)
		os.mkdir(flac_dir)
		count = 0
		for file in tqdm(os.listdir(mp3_dir)):
			if batch_size >=0 and count >= batch_size:
				break		
			if file.split('.')[-1] != "mp3":
				continue
			MP3toFLAC(src=f"{mp3_dir}/{file}", dst=f"{flac_dir}/{file.split('/')[-1][:-3]+'flac'}")
			count += 1

	csv_dir = "resources/bachman_labels.csv"
	assert path.exists(csv_dir)

	print("Processing data...")
	data, labels = cmi.preprocess_data(target=csv_dir, audio_dir=flac_dir, DEBUG=True)
	X_train, X_test, y_train, y_test = train_test_split(data, labels,
	                                                    test_size=0.20)

	embeddings_train, embeddings_test = generate_embeddings(X_train, X_test, feature_model)

	param_grid = {'C': [0.1, 1, 10, 1e2, 1e3],
	              'gamma': [0.001, 0.01, 0.1, 1.0, 10], }

	if classifier == "SVM":
		clf = GridSearchCV(SVC(kernel='rbf', class_weight='balanced',
		                       random_state=random_seed),
		                   param_grid, cv=2)

		svm = clf.fit(embeddings_train.numpy(), y_train)

	elif classifier == "Perceptron":
		clf = GridSearchCV(Perceptron(), param_grid, cv=2)


	y_pred = clf.predict(embeddings_test.numpy())
	print(classification_report(y_test, y_pred))

	from sklearn.metrics import plot_confusion_matrix
	import matplotlib.pyplot as plt
	plot_confusion_matrix(clf, embeddings_test.numpy(), y_test)
	plt.show()
