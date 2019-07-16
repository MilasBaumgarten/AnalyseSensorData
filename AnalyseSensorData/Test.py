import heartpy as hp
from Data import Data
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import UnivariateSpline

min_heart_rate = 90
max_heart_rate = 140
average_heart_rate = 80
min_normalized_value = 0
max_normalized_value = 500
min_beat_delta = 100

def mark_clipping(hrdata, threshold):
    '''function that marks start and end of clipping part
    it detects the start and end of clipping segments and returns them
    
    keyword arguments:
    - data: 1d list or numpy array containing heart rate data
    - threshold: the threshold for clipping, recommended to
                 be a few data points below ADC or sensor max value, 
                 to compensate for signal noise (default 1020)
    
    '''
    clip_binary = np.where(hrdata > threshold)
    clipping_edges = np.where(np.diff(clip_binary) > 1)[1]

    clipping_segments = []

    for i in range(0, len(clipping_edges)):
        if i == 0: #if first clipping segment
            clipping_segments.append((clip_binary[0][0], 
                                      clip_binary[0][clipping_edges[0]]))
        elif i == len(clipping_edges):
            #append last entry
            clipping_segments.append((clip_binary[0][clipping_edges[i]+1],
                                      clip_binary[0][-1]))    
        else:
            clipping_segments.append((clip_binary[0][clipping_edges[i-1] + 1],
                                      clip_binary[0][clipping_edges[i]]))

    return clipping_segments

def interpolate_clipping(hrdata, sample_rate, threshold=1020):
    '''function that interpolates peaks between
    the clipping segments using cubic spline interpolation.
    
    It takes the clipping start +/- 100ms to calculate the spline.
    
    Returns full data array with interpolated segments patched in
    
    keyword arguments:
    data - 1d list or numpy array containing heart rate data
    clipping_segments - list containing tuples of start- and 
                        end-points of clipping segments.
    '''
    clipping_segments = mark_clipping(hrdata, threshold)
    num_datapoints = int(0.1 * sample_rate)
    newx = []
    newy = []
    
    for segment in clipping_segments:
        if segment[0] < num_datapoints: 
            #if clipping is present at start of signal, skip.
            #We cannot interpolate accurately when there is insufficient data prior to clipping segment.
            pass
        else: 
            antecedent = hrdata[segment[0] - num_datapoints : segment[0]]
            consequent = hrdata[segment[1] : segment[1] + num_datapoints]
            segment_data = np.concatenate((antecedent, consequent))
        
            interpdata_x = np.concatenate(([x for x in range(segment[0] - num_datapoints, segment[0])],
                                            [x for x in range(segment[1], segment[1] + num_datapoints)]))
            x_new = np.linspace(segment[0] - num_datapoints,
                                segment[1] + num_datapoints,
                                ((segment[1] - segment[0]) + (2 * num_datapoints)))
        
            interp_func = UnivariateSpline(interpdata_x, segment_data, k=3)
            interp_data = interp_func(x_new)
        
            hrdata[segment[0] - num_datapoints :
                    segment[1] + num_datapoints] = interp_data
       
    return hrdata


# get Data
tracking_file = "TrackingData_2019-05-09-19-19-31"
sensor_file = "SensorData_2019-06-12-16-03-18_Manuel_Heinzig"
train_data = Data( tracking_file + ".txt",
					sensor_file + ".txt",
					min_beat_delta,
					min_normalized_value,
					max_normalized_value,
					min_heart_rate,
					max_heart_rate,
					average_heart_rate)

# mark cutoffs
for i in range(len(train_data.ecg)):
	if (train_data.ecg[i] > 550):
		train_data.ecg[i] = 550

## process Data
sample_rate = hp.get_samplerate_mstimer(train_data.sensor_x)
#filtered = hp.butter_lowpass_filter(train_data.ecg, cutoff=5, sample_rate=100.0, order=3)
data_clipping_fixed = interpolate_clipping(train_data.ecg.copy(), sample_rate, 545)

#enhanced = hp.enhance_peaks(train_data.ecg, iterations=5)

#for key in measures:
#	print(key, " = ", measures[key])

#for key in working_data:
#	print(key)

#for s in working_data["hr"]:
#	print(s)

data = []
for i in range(len(train_data.sensor_x)):
	data.append([train_data.ecg[i], data_clipping_fixed[i]])


plt.figure(1)
plt.subplot(111)
plt.plot(data)

plt.show()


working_data, measures = hp.process(train_data.ecg, sample_rate, interp_clipping=True, interp_threshold=540)

print(measures['bpm']) #returns BPM value
print(measures['rmssd']) # returns RMSSD HRV measure

hp.plotter(working_data, measures)