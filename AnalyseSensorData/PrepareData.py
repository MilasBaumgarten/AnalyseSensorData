from Data import Data
import numpy as np

class PrepareData(object):

	x_full = y_full = x = y_ecg = y_ecg_normalized = y_eda = np.array([])

	def __init__(self, tracking_data_file, sensor_data_file, min_delta):
		self.x_full, self.y_full = Data.extract(tracking_data_file, sensor_data_file)
		self.x = np.squeeze(np.asarray(self.y_full[:,np.array([True, False, False, False])]))
		self.y_ecg = np.squeeze(np.asarray(self.y_full[:,np.array([False, False, True, False])]))
		self.y_eda = np.squeeze(np.asarray(self.y_full[:,np.array([False, True, False, False])]))
		
		y_ecg_delta = self.calc_gradient(self.y_ecg, True)
		self.y_ecg_normalized = self.normalize(y_ecg_delta, min_delta)

	# calculate gradient
	def calc_gradient(self, array, cut = False):
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
	def normalize(self, delta, min_delta):
		last = 0
		normalized = []
		for i in delta:
			if ((np.sign(last) != np.sign(i)) and np.abs(i) >= min_delta):
				normalized.append(500)
			else:
				normalized.append(0)
			last = i
		return normalized