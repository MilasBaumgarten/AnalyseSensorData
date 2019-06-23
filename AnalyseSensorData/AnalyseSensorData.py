#from keras.models import Sequential
#from keras.models import model_from_json
#from keras.layers import Dropout
#from keras import losses
#from keras.layers import Embedding
#from keras.layers import LSTM, Activation, TimeDistributed, Dense

import matplotlib.pyplot as plt

import datetime
import os

import numpy as np

from Data import Data

max_features = 1024

min_heart_rate = 60
max_heart_rate = 180
average_heart_rate = 80
min_normalized_value = 0
max_normalized_value = 500
min_beat_delta = 40

tracking_file = "TrackingData_2019-05-09-19-19-31"
#sensor_file = "SensorData_2019-06-12-13-58-08_Bastian"
#sensor_file = "SensorData_2019-06-12-14-09-52_Tobi"
#sensor_file = "SensorData_2019-06-12-14-23-59_Simon"
##sensor_file = "SensorData_2019-06-12-14-43-52_Isa"
#sensor_file = "SensorData_2019-06-12-14-58-58_Juliane"
##sensor_file = "SensorData_2019-06-12-15-14-22_Mareike_ohne_Telemetrie"
#sensor_file = "SensorData_2019-06-12-15-40-17_Sebastian"
#sensor_file = "SensorData_2019-06-12-15-51-06_Mareike"
sensor_file = "SensorData_2019-06-12-16-03-18_Manuel_Heinzig"
#sensor_file = "SensorData_2019-06-12-16-17-01_Milas"


#loss_function = losses.binary_crossentropy
#optimizer_function = "rmsprop"
#metrics_function = ["accuracy"]

#directory = datetime.datetime.now().strftime("%Y_%m_%d-%H_%M")
#filename = directory + "/generation"

#if not os.path.exists(directory):
#	os.makedirs(directory)


train_data = Data( tracking_file + ".txt",
					sensor_file + ".txt",
					min_beat_delta,
					min_normalized_value,
					max_normalized_value,
					min_heart_rate,
					max_heart_rate,
					average_heart_rate)

#test_data = Data( "TrackingData_2019-05-09-19-13-42_cleaned.txt",
#					"SensorData_2019-05-09-19-13-42_cleaned.txt",
#					min_beat_delta,
#					min_normalized_value,
#					max_normalized_value,
#					min_heart_rate,
#					max_heart_rate,
#					average_heart_rate)

# combine arrays training
in_train, out_train = train_data.prepare_data_for_keras()
#in_test, out_test = test_data.prepare_data_for_keras()

##############
# LSTM Stuff #
##############

# prepare data shape for the LSTM
in_train = np.reshape(in_train, (in_train.shape[0], in_train.shape[1], 1))
#in_test = np.reshape(in_test, (in_test.shape[0], in_test.shape[1], 1))

# TODO:	create LSTM
#		insert data into LSTM
#		train LSTM

#################
# save the data #
#################
directory = sensor_file
file = directory + "/data.txt"

if not os.path.exists(directory):
	os.makedirs(directory)

#if (os.path.isfile(file)):
#	os.remove(file)

print("save sensor data to file")
train_data.save_to_file(file, 
						"Time\tECG\tECG_Normalized\tHeart_Rate\tEDA\n",
						train_data.sensor_x,train_data.ecg, train_data.ecg_normalized, train_data.heart_rate, train_data.eda)

#################
# plot the data #
#################
plt.figure(1)
plt.subplot(221)
ecg_plot, = plt.plot(train_data.sensor_x, np.array(train_data.ecg), label="ECG Data")
plt.legend(handles=[ecg_plot], loc=1)

plt.subplot(222)
eda_plot, = plt.plot(train_data.sensor_x, np.array(train_data.eda), label="EDA Data")
plt.legend(handles=[eda_plot], loc=1)

plt.subplot(223)
ecg_normalized_plot, = plt.plot(train_data.sensor_x, np.array(train_data.ecg_normalized), label="ECG normalized")
plt.legend(handles=[ecg_normalized_plot], loc=1)

plt.subplot(224)
heart_rate_plot, = plt.plot(train_data.sensor_x, np.array(train_data.heart_rate), label="ECG Frequency")
plt.legend(handles=[heart_rate_plot], loc=1)

plt.show()