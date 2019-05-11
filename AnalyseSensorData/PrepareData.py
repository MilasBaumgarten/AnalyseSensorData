from Data import Data
import numpy as np

class PrepareData(object):

	x_full = y_full = x = y_ecg = y_ecg_normalized = y_eda = np.array([])
	beat_delay = []

	def __init__(self, tracking_data_file, sensor_data_file, min_delta, min_val, max_val, min_heart_beat_delay, max_heart_beat_delay, average_heart_beat_delay):
		self.x_full, self.y_full = Data.extract(tracking_data_file, sensor_data_file)
		self.x = np.squeeze(np.asarray(self.y_full[:,np.array([True, False, False, False])]))
		self.y_ecg = np.squeeze(np.asarray(self.y_full[:,np.array([False, False, True, False])]))
		self.y_eda = np.squeeze(np.asarray(self.y_full[:,np.array([False, True, False, False])]))
		
		y_ecg_delta = self.calc_gradient(self.y_ecg, True)
		self.y_ecg_normalized = self.normalize(y_ecg_delta, min_delta, min_val, max_val)

		for i in range(0, 3):
			print(self.beat_delay)
			self.calculate_missed_heart_beats(min_val, max_val, min_heart_beat_delay, max_heart_beat_delay, average_heart_beat_delay)
			print("-------------------------------")
		print(self.beat_delay)

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
		for i in range(0, len(self.y_ecg_normalized)):
			if (self.y_ecg_normalized[i] == max_val):
				index_delta = (i - last)

				if (index_delta < min_delay):
					self.y_ecg_normalized[last] = min_val
				else :
					if (index_delta > max_delay):
						# add missed heart beats
						missed_beats = index_delta // average_delay
						for j in range(1, missed_beats):
							#self.y_ecg_normalized[index_delta + int(np.floor((j / missed_beats) * index_delta))] = max_val
							self.y_ecg_normalized[last + int(np.floor((j / missed_beats) * index_delta))] = max_val
							self.beat_delay.append(int(np.floor((1 / missed_beats) * index_delta)))
					else:
						self.beat_delay.append(index_delta)
					last = i