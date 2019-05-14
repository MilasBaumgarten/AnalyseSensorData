import matplotlib.pyplot as plt
import numpy as np
from Data import Data

min_heart_beat_delay = 20
max_heart_beat_delay = 35
average_heart_beate_delay = 25
min_normalized_value = 0
max_normalized_value = 1
min_beat_delta = 40

data = Data( "TrackingData_2019-05-09-19-13-42.txt",
					"SensorData_2019-05-09-19-13-42.txt",
					min_beat_delta,
					min_normalized_value,
					max_normalized_value,
					min_heart_beat_delay,
					max_heart_beat_delay,
					average_heart_beate_delay)

# combine arrays for the graph
combined = []
print(len(data.ecg_synchronized))
print(len(data.tracking_x))
for i in range (0, len(data.tracking_x)):
	combined.append([data.ecg_synchronized[i], data.HMD[i].position.x])

combined_sensor = []
for i in range (0, len(data.sensor_x)):
	combined_sensor.append([data.ecg[i], data.ecg_normalized[i]])

combined_HMD_pos = []
for i in range (0, len(data.tracking_x)):
	combined_HMD_pos.append([data.HMD[i].position.x, data.HMD[i].position.y, data.HMD[i].position.z])

combined_HMD_rot = []
for i in range (0, len(data.tracking_x)):
	combined_HMD_rot.append([data.HMD[i].rotation.x, data.HMD[i].rotation.y, data.HMD[i].rotation.z, data.HMD[i].rotation.w])

print(len(data.sensor_x), " - ", len(data.tracking_x))

# draw graph
plt.plot(data.tracking_x, combined)
#plt.figure(1)
#plt.subplot(221)
#plt.ylabel("ecg")
#plt.plot(data.sensor_x, combined_sensor)

#plt.subplot(222)
#plt.ylabel("delay")
#plt.plot(data.beat_delay)

#plt.subplot(223)
#plt.ylabel("HMD Position")
#plt.plot(data.tracking_x, combined_HMD_pos)

#plt.subplot(224)
#plt.ylabel("HMD Rotation")
#plt.plot(data.tracking_x, combined_HMD_rot)

plt.show()