import matplotlib.pyplot as plt

import datetime

import numpy as np

from Data import Data

max_features = 1024

min_heart_rate = 90
max_heart_rate = 140
average_heart_rate = 80
min_normalized_value = 0
max_normalized_value = 500
min_beat_delta = 100

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

train_data = Data( tracking_file + ".txt",
					sensor_file + ".txt",
					min_beat_delta,
					min_normalized_value,
					max_normalized_value,
					min_heart_rate,
					max_heart_rate,
					average_heart_rate)

# combine arrays training
in_train, out_train = train_data.prepare_data_for_keras(False)

# prepare data shape for the LSTM
in_train = np.reshape(in_train, (in_train.shape[0], in_train.shape[1], 1))

# TODO:	create LSTM
#		insert data into LSTM
#		train LSTM

delay = []
last = 0
for i in range(0, len(train_data.sensor_x)):
	if (train_data.ecg_normalized[i] == max_normalized_value):
		time = train_data.sensor_x[i] - train_data.sensor_x[last]
		delay.append(time)
		print(time)
		last = i

#################
# save the data #
#################
train_data.save_data(sensor_file)

#################
# plot the data #
#################
plt.figure(1)
#plt.subplot(221)
#ecg_plot, = plt.plot(train_data.sensor_x, np.array(train_data.ecg), label="ECG Data")
#plt.legend(handles=[ecg_plot], loc=1)

#plt.subplot(222)
#eda_plot, = plt.plot(train_data.sensor_x, np.array(train_data.eda), label="EDA Data")
#plt.legend(handles=[eda_plot], loc=1)

#plt.subplot(223)
#ecg_normalized_plot, = plt.plot(train_data.sensor_x, np.array(train_data.ecg_normalized), label="ECG normalized")
#plt.legend(handles=[ecg_normalized_plot], loc=1)

#plt.subplot(224)
heart_rate_plot, = plt.plot(delay, label="ECG Frequency")
#heart_rate_plot, = plt.plot(train_data.sensor_x, np.array(train_data.heart_rate), label="ECG Frequency")
plt.legend(handles=[heart_rate_plot], loc=1)

plt.show()