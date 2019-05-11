import numpy as np
import sys

class Data(object):
	def __init__(self):
		print("created")

	# Source: "The Best Way to Prepare a Dataset Easily" - Siraj Raval - https://www.youtube.com/watch?v=0xVqLJe9_CY&t=437s
	@staticmethod
	def extract(tracking_filename, sensor_filename):
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

	@staticmethod
	def generate_dummy_data():
		x_train = []	# input
		y_train = []	# target output
		for i in range(0, 1000):
			x_train.append(np.random.random())
			y_train.append(np.sin(x_train[i]))

		x_train = np.array(x_train)
		y_train = np.array(y_train)

		# generate dummy data for final test
		x_test = []
		y_test = []

		for i in range(0, 100):
			x_test.append(np.random.random())
			y_test.append(np.sin(x_test[i]))

		return x_train, y_train, x_test, y_test