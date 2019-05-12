import numpy as np
import sys

class Data(object):
	x_full = y_full = x = y_ecg = y_ecg_normalized = y_ecg_delta = y_eda = np.array([])
	beat_delay = []

	def __init__(self, tracking_data_file, sensor_data_file, min_delta, min_val, max_val, min_heart_beat_delay, max_heart_beat_delay, average_heart_beat_delay):
		# get raw data and separate into specific arrays
		self.x_full, self.y_full = self.extract(tracking_data_file, sensor_data_file)
		self.x = np.squeeze(np.asarray(self.y_full[:,np.array([True, False, False, False])]))
		self.y_ecg = np.squeeze(np.asarray(self.y_full[:,np.array([False, False, True, False])]))
		self.y_eda = np.squeeze(np.asarray(self.y_full[:,np.array([False, True, False, False])]))
		
		# calculate gradient and normalize ecg data
		self.y_ecg_delta = self.calc_gradient(self.y_ecg, True)
		self.y_ecg_normalized = self.normalize(self.y_ecg_delta, min_delta, min_val, max_val)

		# try to fix measurment errors by inserting missed heartbeats and deleting duplicate heart beats
		for i in range(0, 1):
			self.calculate_missed_heart_beats(min_val, max_val, min_heart_beat_delay, max_heart_beat_delay, average_heart_beat_delay)

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
		# TODO: synchronize sensor data and tracking data

		# add player tracking data values to feature vectors
		for line in tracking_lines:
			try:
				line = line.replace("\t", " ")
				line = line.replace(",", ".")
				row = line.split(" ")
				# add all relevant elements to the feature vectors (from a to (exclusive) b)
				fvecs.append([float(x) for x in row[1:22]])
			except:
				print("Error with line: " + line)

		# add sensor data to labels
		for line in sensor_lines:
			line = line.replace(";", "")
			row = line.split(" ")
			labels.append([float(x) for x in row[0:4]])

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
		self.beat_delay.clear()
		last = 0
		for i in range(0, len(self.y_ecg_normalized)):
			# start check when a heart beat was found
			if (self.y_ecg_normalized[i] == max_val):
				index_delta = (i - last)
				# remove a heart beat if it seems to be too fast aka duplicate
				if (index_delta < min_delay):
					# check which gradient was higher
					if (self.y_ecg_delta[last] < self.y_ecg_delta[i]):
						self.y_ecg_normalized[last] = min_val
					else:
						self.y_ecg_normalized[i] = min_val
				else :
					# insert missed heart beats
					if (index_delta > max_delay):
						missed_beats = index_delta // average_delay
						for j in range(1, missed_beats):
							self.y_ecg_normalized[last + int(np.floor((j / missed_beats) * index_delta))] = max_val
							self.beat_delay.append(int(np.floor((1 / missed_beats) * index_delta)))
					else:
						self.beat_delay.append(index_delta)
					last = i