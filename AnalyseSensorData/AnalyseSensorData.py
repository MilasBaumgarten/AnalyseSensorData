from keras.models import Sequential
from keras.models import model_from_json
from keras.layers import Dropout
from keras import losses
from keras.layers import Embedding
from keras.layers import LSTM, Activation, TimeDistributed, Dense

import matplotlib.pyplot as plt

import datetime
import os

import numpy as np

from Data import Data

max_features = 1024

min_heart_beat_delay = 20
max_heart_beat_delay = 35
average_heart_beate_delay = 25
min_normalized_value = 1
max_normalized_value = 2
min_beat_delta = 40

# Loss Result (40 Epochs)	Goal ~ 390
# mean_squared_error:				const. 284.1575
# mean_absolute_error:				262.34/265/197.95706
# mean_absolute_percentage_error:	221/242.85115/184.94001
# mean_squared_logarithmic_error:	2:1 276.6/192.2
# squared_hinge:					~1, ecg signal (incorrect but periodic (too fast))
# hinge:							~2, ecg signal (incorrect but periodic (too fast))
# categorical_hinge:				~1
# logcosh:							269.79593/270.05362/202.64323
# categorical_crossentropy:			nan
# sparse_categorical_crossentropy:	error - dense layer shape
# binary_crossentropy:				2:1 ~5.05/1.5825226
# kullback_leibler_divergence:		2:1 ~5/2.087649
# poisson:							2:1 288.13986/197.53024
# cosine_proximity:					2.2061598/2.46/2.539

# Hidden: ReLU
# Output: Softmax - Classification
#		  Linear  - Regression
# 100 Generations, 10 Epochs, Batch Size: 32
# ecg normalized, eda notmalization: x / 1024
# binary crossentropy: ~0.4 (goal ~0.2), no heart beat

loss_function = losses.binary_crossentropy
optimizer_function = "rmsprop"
metrics_function = ["accuracy"]

directory = datetime.datetime.now().strftime("%Y_%m_%d-%H_%M")
filename = directory + "/generation"

if not os.path.exists(directory):
	os.makedirs(directory)

def save_model(iteration):
	# source: https://machinelearningmastery.com/save-load-keras-deep-learning-models/
	# source: https://keras.io/getting-started/faq/#how-can-i-save-a-keras-model
	model_json = model.to_json()

	# with is similar to try/ catch but better(?)
	with open(filename + str(iteration) + ".json", "w") as json_file:
		json_file.write(model_json)
	model.save_weights(filename + str(iteration) + ".h5")

def load_model(iteration):
	# load json and create model
	json_file = open("model" + str(iteration) + ".json", "r")
	loaded_model_json = json_file.read()
	json_file.close()
	loaded_model = model_from_json(loaded_model_json)

	# load weights into new model
	loaded_model.load_weights("model" + str(iteration) + ".h5")
	print("Loaded model from disk")

	# evaluate loaded model on test data
	loaded_model.compile(loss=loss_function, optimizer=optimizer_function, metrics=metrics_function)
	score = loaded_model.evaluate(in_test, out_test)
	print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))

	return loaded_model

train_data = Data( "TrackingData_2019-05-09-19-19-31.txt",
					"SensorData_2019-05-09-19-19-31.txt",
					min_beat_delta,
					min_normalized_value,
					max_normalized_value,
					min_heart_beat_delay,
					max_heart_beat_delay,
					average_heart_beate_delay)

test_data = Data( "TrackingData_2019-05-09-19-13-42_cleaned.txt",
					"SensorData_2019-05-09-19-13-42_cleaned.txt",
					min_beat_delta,
					min_normalized_value,
					max_normalized_value,
					min_heart_beat_delay,
					max_heart_beat_delay,
					average_heart_beate_delay)

# combine arrays training
in_train, out_train = train_data.prepare_data_for_keras()
in_test, out_test = test_data.prepare_data_for_keras()

print(in_train)
print(out_train)

##############
# LSTM Stuff #
##############

# prepare data shape for the LSTM
in_train = np.reshape(in_train, (in_train.shape[0], in_train.shape[1], 1))
in_test = np.reshape(in_test, (in_test.shape[0], in_test.shape[1], 1))

## create model
#model = Sequential()
##model.add(Dense(2, input_shape=(1,21)))
##model.add(Embedding(max_features, output_dim=256))
##model.add(LSTM(256, input_shape=(1, 21), activation="relu", recurrent_activation="hard_sigmoid", return_sequences=True))
#model.add(LSTM(256, input_shape=(1, 21), return_sequences=True))
#model.add(LSTM(256))
##model.add(TimeDistributed(Dense(21)))																				# what does this do?
#model.add(Activation("softmax"))
##model.add(LSTM(128, activation="relu", recurrent_activation="hard_sigmoid"))
#model.add(Dropout(0.5))
#model.add(Dense(2, activation="relu"))

## compile model
#model.compile(loss=loss_function, optimizer=optimizer_function, metrics=metrics_function)

#print(model.input_shape)
#print(model.output_shape)

predictions = model.predict(in_test)
# train model
#for i in range(0, 1):
#	# save model every n epochs
#	print("-----------------------")
#	print (i + 1, "/", 4)
#	print("-----------------------")
#	model.fit(in_train, out_train, batch_size=32, epochs=10)
#	save_model(i)

## evaluate model
#score = model.evaluate(in_test, out_test, batch_size=16)
#print("%s: %.2f%%" % (model.metrics_names[1], score[1]*100))

#predictions = model.predict(in_test)

combined_ecg = []
combined_eda = []
for i in range (0, 500):
	combined_ecg.append([out_test[i][0]])
	combined_eda.append([out_test[i][1]])

plt.figure(1)
plt.subplot(221)
plt.plot(combined_ecg)

plt.subplot(222)
plt.plot(combined_eda)

print(train_data.sensor_x)
print(np.array(train_data.ecg_normalized))

plt.subplot(223)
plt.plot(train_data.sensor_x, np.array(train_data.ecg_normalized))

plt.show()

#for i in range(0, len(in_test)):
#for i in range(0, 500):
#	print(predictions[i], " - ", out_test[i])

#for i in range(0, 2):
#	load_model(i)
