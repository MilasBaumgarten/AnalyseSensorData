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
	beat_delay = []
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
		print(self.HMD[0].position.x)
		
		# calculate gradient and normalize ecg data
		self.ecg_delta = self.calc_gradient(self.ecg, True)
		self.ecg_normalized = self.normalize(self.ecg_delta, min_delta, min_val, max_val)

		# try to fix measurment errors by inserting missed heartbeats and deleting
		# duplicate heart beats
		for i in range(0, 1):
			self.calculate_missed_heart_beats(self.LOW, self.HIGH, min_heart_beat_delay, max_heart_beat_delay, average_heart_beat_delay)

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
		# TODO: synchronize sensor data and tracking data

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
		self.beat_delay.clear()
		last = 0
		for i in range(0, len(self.ecg_normalized)):
			# start check when a heart beat was found
			if (self.ecg_normalized[i] == max_val):
				index_delta = (i - last)
				# remove a heart beat if it seems to be too fast aka duplicate
				if (index_delta < min_delay):
					# check which gradient was higher
					if (self.ecg_delta[last] < self.ecg_delta[i]):
						self.ecg_normalized[last] = min_val
					else:
						self.ecg_normalized[i] = min_val
				else :
					# insert missed heart beats
					if (index_delta > max_delay):
						missed_beats = index_delta // average_delay
						for j in range(1, missed_beats):
							self.ecg_normalized[last + int(np.floor((j / missed_beats) * index_delta))] = max_val
							self.beat_delay.append(int(np.floor((1 / missed_beats) * index_delta)))
					else:
						self.beat_delay.append(index_delta)
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
			percentage = 0
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

# TODO:
#	synchronization: prioritise heart beats