import matplotlib.pyplot as plt
import numpy as np
from Data import Data
from PrepareData import PrepareData

min_heart_beat_delay = 15
max_heart_beat_delay = 45
average_heart_beate_delay = 25
min_normalized_value = 0
max_normalized_value = 500
min_beat_delta = 50

data = PrepareData( "TrackingData_2019-05-09-19-13-42.txt",
					"SensorData_2019-05-09-19-13-42.txt",
					min_beat_delta,
					min_normalized_value,
					max_normalized_value,
					min_heart_beat_delay,
					max_heart_beat_delay,
					average_heart_beate_delay)

print(np.average(data.beat_delay))


# combine arrays for the graph
combined = []
for i in range (0, len(data.y_ecg)):
	combined.append([data.y_ecg[i], data.y_ecg_normalized[i]])

# draw graph
plt.ylabel("ecg")
plt.plot(data.x, combined)
#plt.plot(data.beat_delay)

plt.show()