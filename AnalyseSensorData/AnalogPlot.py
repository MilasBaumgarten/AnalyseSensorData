# source https://codereview.stackexchange.com/questions/155905/time-scrolling-interface-in-matplotlib

import numpy as np
from collections import deque

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Button, Slider
from Data import Data


class AnalogPlot:

	def __init__(self, data, display_len):
		self.buff = deque(np.zeros(display_len))
		self.display_len = display_len
		self.data = data

		# set up the plot
		self.fig = plt.figure()
		self.ax = self.fig.add_subplot(111, xlim=(0, t_max), ylim=(0, 1.5))
		self.ax.set_xticks((0, t_max))
		self.lines = self.ax.plot([], [])

		# setup the animation
		self.cur_frame = 0
		self.anim = animation.FuncAnimation(self.fig, self._update,
											interval=1.0)

		# setup the animation control
		self.anim_running = True

	def _add_to_buff(self, buf, val):
		if len(buf) < self.display_len:
			buf.appendLeft(val)
		else:
			buf.popleft()
			buf.append(val)

	def _update(self, frame):
		frame = self.cur_frame
		self._add_to_buff(self.buff, self.data[frame:frame+1])
		self.lines[0].set_data(range(self.display_len), self.buff)

		self.ax.set_xticklabels((str(frame), str(frame+self.display_len)))

		self.time_slider.eventson = False
		self.time_slider.set_val(frame)
		self.time_slider.eventson = True

		self.cur_frame += 1

		return self.lines

	def _pause(self, event):
		if self.anim_running:
			self.anim.event_source.stop()
			self.anim_running = False
		else:
			self.anim.event_source.start()
			self.anim_running = True

	def _reset(self, event):
		self._set_val(0)

	def _set_val(self, frame=0):
		frame = int(frame)
		self.cur_frame = frame
		new_start = frame - self.display_len
		if new_start >= 0:
			self.buff = deque(self.data[new_start:frame])
		else:
			self.buff = deque(np.concatenate((np.zeros(np.abs(new_start)),
												self.data[:frame])))

		self.anim.event_source.stop()
		self.anim = animation.FuncAnimation(self.fig, self._update,
											interval=1.0)
		self.anim_running = True

	def animate(self):
		pause_ax = self.fig.add_axes((0.7, 0.025, 0.1, 0.04))
		pause_button = Button(pause_ax, 'pause', hovercolor='0.975')
		pause_button.on_clicked(self._pause)

		reset_ax = self.fig.add_axes((0.8, 0.025, 0.1, 0.04))
		reset_button = Button(reset_ax, 'reset', hovercolor='0.975')
		reset_button.on_clicked(self._reset)

		slider_ax = self.fig.add_axes((0.1, 0.025, 0.5, 0.04))
		self.time_slider = Slider(slider_ax, label='Time',
									valmin=0, valmax=self.data.shape[0],
									valinit=0.0)

		self.time_slider.on_changed(self._set_val)

		plt.show()


t_max = 300

in_train, out_train = Data.extract("TrackingData_2019-05-09-19-13-42.txt", "SensorData_2019-05-09-19-13-42.txt")

delay = out_train[0, 0]

x = np.squeeze(np.asarray(out_train[:,np.array([True, False, False, False])]))
y_eda = np.squeeze(np.asarray(out_train[:,np.array([False, True, False, False])]))
y_ecg = np.squeeze(np.asarray(out_train[:,np.array([False, False, True, False])]))

# normalize time
for i in x:
	i -= delay

# clean ecg
min_heart_beat_delta = 50
delta = 0
last = 0

y_ecg_normalized = []

for i in y_ecg:
	delta = i - last
	last = i
	if (delta >= min_heart_beat_delta):
		y_ecg_normalized.append(1)
	else:
		y_ecg_normalized.append(0)

print(y_ecg_normalized)

analog_plot = AnalogPlot(np.array(y_ecg_normalized), t_max)
analog_plot.animate()