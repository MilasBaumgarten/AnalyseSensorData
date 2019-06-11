from dataclasses import dataclass
import numpy as np
import sys

@dataclass
class Position:
	x : float
	y : float
	z : float

@dataclass
class Rotation:
	x : float
	y : float
	z : float
	w : float

@dataclass
class Transform:
	position : Position
	rotation : Rotation

class Data(object):
	HMD = RightTracker = LeftTracker = []

	HIGH = 0
	LOW = 0

	def __init__(self, tracking_data_file, sensor_data_file, min_delta, min_val, max_val, min_heart_beat_delay, max_heart_beat_delay, average_heart_beat_delay):
		self.HIGH = max_val
		self.LOW = min_val

		# get raw data and separate into specific arrays
		tracking_full, sensor_full = self.extract(tracking_data_file, sensor_data_file)
		self.sensor_x = np.squeeze(np.asarray(sensor_full[:,np.array([True, False, False, False])]))
		self.ecg = np.squeeze(np.asarray(sensor_full[:,np.array([False, False, True, False])]))
		self.eda = np.squeeze(np.asarray(sensor_full[:,np.array([False, True, False, False])]))
		
		# format tracking data
		tracking_x =  []
		for line in tracking_full:
			tracking_x.append(line[0,0])
			self.HMD.append(Transform(Position(line[0, 1], line[0, 2], line[0, 3]), Rotation(line[0, 4], line[0, 5], line[0, 6], line[0, 7])))
			self.RightTracker.append(Transform(Position(line[0, 8], line[0, 9], line[0,10]), Rotation(line[0,11], line[0,12], line[0,13], line[0,14])))
			self.LeftTracker.append(Transform(Position(line[0,15], line[0,16], line[0,17]), Rotation(line[0,18], line[0,19], line[0,20], line[0,21])))

		self.tracking_x = np.array(tracking_x)
		
		# calculate gradient and normalize ecg data
		self.ecg_delta = self.calc_gradient(self.ecg, True)
		self.ecg_normalized = self.normalize(self.ecg_delta, min_delta, min_val, max_val)

		# try to fix measurment errors by inserting missed heartbeats and deleting
		# duplicate heart beats
		for i in range(0, 1):
			self.calculate_missed_heart_beats(self.LOW, self.HIGH, 60000 / max_heart_beat_delay,
																   60000 / min_heart_beat_delay,
																   60000 / average_heart_beat_delay)

		self.synchronize()

	# Source: "The Best Way to Prepare a Dataset Easily" - Siraj Raval - https://www.youtube.com/watch?v=0xVqLJe9_CY&t=437s
	def extract(self, tracking_filename, sensor_filename):
		# arrays to hod the labels and the feature vectors
		labels = []
		fvecs = []

		tracking = open(tracking_filename, "r").read()
		tracking_lines = tracking.split("\n")

		sensor = open(sensor_filename, "r").read()
		sensor_lines = sensor.split("\n")

		# TODO: make sensor_lines and tracking_lines equal length

		# add player tracking data values to feature vectors
		for line in tracking_lines:
			try:
				line = line.replace("\t", " ")
				line = line.replace(",", ".")
				row = line.split(" ")
				# add all relevant elements to the feature vectors (from a to (exclusive)
				# b)
				fvecs.append([float(x) for x in row[0:22]])
			except:
				print("Error with line: " + line)

		# add sensor data to labels
		for line in sensor_lines:
			line = line.replace(";", "")
			row = line.split(" ")
			labels.append([float(x) for x in row[0:4]])

		# remove time delay from sensor data
		sensor_delay = labels[0][0]
		for date in labels:
			date[0] -= sensor_delay

		# convert the features into a numpy float matrix
		fvecs_np = np.matrix(fvecs).astype(np.float32)

		# convert the labels into a numpy array
		labels_np = np.array(labels).astype(np.float32)

		#return a pair of the feature matrix and the label matrix
		return fvecs_np, labels_np

	# calculate gradient
	def calc_gradient(self, array, cut):
		delta = 0
		last = 0
		gradient = []
		for i in array:
			delta = i - last
			last = i
			if (cut and delta < 0):
				delta = 0
			gradient.append(np.abs(delta))
		return gradient

	# normalize values
	def normalize(self, delta, min_delta, min_val, max_val):
		last = 0
		normalized = []
		for i in delta:
			if ((np.sign(last) != np.sign(i)) and np.abs(i) >= min_delta):
				normalized.append(max_val)
			else:
				normalized.append(min_val)
			last = i
		return normalized

	# calculate time between heart (suspected) beats
	def calculate_missed_heart_beats(self, min_val, max_val, min_delay, max_delay, average_delay):
		last = 0
		for i in range(0, len(self.ecg_normalized)):
			# start check when a heart beat was found
			if (self.ecg_normalized[i] == max_val):
				delta_t = (self.sensor_x[i] - self.sensor_x[last])
				index_delta = (i - last)
				# remove a heart beat if it seems to be too fast aka duplicate
				if (delta_t < min_delay):
					# check which gradient was higher
					if (self.ecg_delta[last] < self.ecg_delta[i]):
						self.ecg_normalized[last] = min_val
						last = i
					else:
						self.ecg_normalized[i] = min_val
				else :
					# insert missed heart beats
					if (delta_t > max_delay):
						missed_beats = int(np.round(delta_t // average_delay))
						for j in range(1, missed_beats + 1):
							self.ecg_normalized[last + int(np.round((j / (missed_beats + 1)) * index_delta))] = max_val
					last = i
		return

	# map sensor data onto tracking data
	def synchronize(self):
		ecg_synchronized = []
		eda_synchronized = []
		index_sensor = 0
		high_recognized = False

		for i in range (0, len(self.tracking_x) - 1):
			# percentage = percentage at which point the next tracking date is in comparison to the current sensor date and the next one
			percentage = (self.sensor_x[index_sensor + 1] - self.sensor_x[index_sensor]) / (self.tracking_x[i] - self.sensor_x[index_sensor])
			if ((percentage < 0.5 and self.ecg_normalized[index_sensor] == self.HIGH) or not (high_recognized)):
				ecg_synchronized.append(self.HIGH)
				high_recognized = True
			else:
				ecg_synchronized.append(self.LOW)

			# sensor_data(tracking_index_x) = sensor_data(last_sensor_index) + percentage * sensor_date_delta_to_next_sensor_date
			eda_synchronized.append(self.eda[index_sensor] + percentage * (self.eda[index_sensor + 1] - self.eda[index_sensor]))

			while not (self.sensor_x[index_sensor + 1] > self.tracking_x[i + 1]):
				if (self.ecg_normalized[index_sensor] == self.HIGH) and not (high_recognized):
					high_recognized = False
				index_sensor += 1

		# add last data
		if ((percentage < 0.5 and self.ecg_normalized[index_sensor] == self.HIGH) or not (high_recognized)):
			ecg_synchronized.append(self.HIGH)
		else:
			ecg_synchronized.append(self.LOW)
		eda_synchronized.append(self.eda[index_sensor] + percentage * (self.eda[index_sensor + 1] - self.eda[index_sensor]))

		self.ecg_synchronized = np.array(ecg_synchronized)
		self.eda_synchronized = np.array(eda_synchronized)

	# prepare data for Keras
	def prepare_data_for_keras(self):
		# combine arrays training
		in_data = []
		out_data = []

		for i in range (0, len(self.tracking_x)):
			in_data.append([self.HMD[i].position.x, self.HMD[i].position.y, self.HMD[i].position.z, self.HMD[i].rotation.x, self.HMD[i].rotation.y, self.HMD[i].rotation.z, self.HMD[i].rotation.w,
							 self.LeftTracker[i].position.x, self.LeftTracker[i].position.y, self.LeftTracker[i].position.z, self.LeftTracker[i].rotation.x, self.LeftTracker[i].rotation.y, self.LeftTracker[i].rotation.z, self.LeftTracker[i].rotation.w,
							 self.RightTracker[i].position.x, self.RightTracker[i].position.y, self.RightTracker[i].position.z, self.RightTracker[i].rotation.x, self.RightTracker[i].rotation.y, self.RightTracker[i].rotation.z, self.RightTracker[i].rotation.w])

		for i in range (0, len(self.tracking_x)):
			out_data.append([self.ecg_synchronized[i], self.eda_synchronized[i] / 1024])
		
		return (np.array(in_data), np.array(out_data))
