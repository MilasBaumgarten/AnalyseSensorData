import matplotlib.pyplot as plt
import numpy as np
from Data import Data

in_train, out_train = Data.extract("TrackingData_2019-05-09-19-13-42.txt", "SensorData_2019-05-09-19-13-42.txt")

delay = out_train[0, 0]

x = np.squeeze(np.asarray(out_train[:,np.array([True, False, False, False])]))
y_eda = np.squeeze(np.asarray(out_train[:,np.array([False, True, False, False])]))
y_ecg = np.squeeze(np.asarray(out_train[:,np.array([False, False, True, False])]))

# normalize time
for i in x:
	i -= delay

plt.figure(1)
plt.subplot(211)
plt.ylabel("ecg")
plt.plot(x, y_ecg)

plt.subplot(212)
plt.ylabel("eda")
plt.plot(x, y_eda)

plt.show()