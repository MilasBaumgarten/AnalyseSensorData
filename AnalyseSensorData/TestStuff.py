import matplotlib.pyplot as plt
import numpy as np
from Data import Data
from PrepareData import PrepareData

data = PrepareData("TrackingData_2019-05-09-19-13-42.txt", "SensorData_2019-05-09-19-13-42.txt", 50)

# calculate time between heart (suspected) beats
beat_delay = []
last = 0
for i in range(0, len(data.y_ecg_normalized)):
	if (data.y_ecg_normalized[i] > 0):
		index_delta = (i-last)
		if (index_delta <= 15):
			data.y_ecg_normalized[index_delta] = 0
		else :
			if (index_delta >= 40):
				# add missed heart beats
				missed_beats = index_delta // 25
				print("{}/{} = {} -> {}".format(last, i, index_delta, missed_beats))
				for j in range(1, missed_beats):
					data.y_ecg_normalized[index_delta + int(np.floor((j / missed_beats) * index_delta))] = 500
					beat_delay.append(index_delta + int(np.floor((j / missed_beats) * index_delta)))
			else:
				beat_delay.append(index_delta)
			last = i

beat_average = np.average(beat_delay)

print(beat_average)
print(beat_delay)


# combine arrays for the graph
combined = []
for i in range (0, len(data.y_ecg)):
	combined.append([data.y_ecg[i], data.y_ecg_normalized[i]])

# draw graph
plt.ylabel("ecg")
#plt.plot(data.x, combined)
plt.plot(beat_delay)

plt.show()