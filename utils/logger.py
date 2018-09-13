import numpy as np
import re

class Measure:
	measurements = []
	measure_name = ''
	measure_normalizer = None
	legend = None

	def __init__(self, measure_name):
		self.measure_name = measure_name
		self.measurements = []

	def reset(self):
		self.measurements = []

	def add_measurement(self, measurement, epoch, iteration=None):
		while len(self.measurements)-1 < epoch:
			self.measurements.append(None)
		if iteration is None:
			self.measurements[epoch] = measurement
		else:
			if self.measurements[epoch] is None:
				self.measurements[epoch] = []
			while len(self.measurements[epoch])-1 < iteration:
				self.measurements[epoch].append(None)
			self.measurements[epoch][iteration] = measurement

	def mean_epoch(self, epoch=-1):
		vals = self.measurements[epoch]
		if type(vals) == list:
			if self.measure_normalizer is None:
				return np.array(vals).mean()
			else:
				return np.array(vals).sum()/self.measure_normalizer
		else:
			return vals

	def sum_epoch(self, epoch=-1):
		vals = self.measurements[epoch]
		if type(vals) == list:
			return np.array(vals).sum()
		else:
			return vals

	def visualize_epoch(self, visdom, epoch=-1):
		vals = self.measurements[epoch]
		assert type(vals) == list, 'The requested epoch visualization is not a list! %s'%self.measure_name

		title = self.measure_name
		win   = self.measure_name
		X = np.arange(len(self.measurements[epoch]));
		Y = np.array(self.measurements[epoch])
		if epoch == -1:
			epoch = len(self.measurements)-1
		visdom.line(X=X, Y=Y,
			opts = dict(title = 'Epoch %d - %s'%(epoch, title)), win=win) 

	def visualize_all_epochs(self, visdom):
		"""
			When you've been loggin a vector/iteration you'd want to visualize
			them all this way.
		"""
		vals = self.measurements
		assert type(vals) == list, 'The requested epoch visualization is not a list! %s'%self.measure_name

		title = self.measure_name
		win   = self.measure_name
		X = np.arange(len(self.measurements))
		Y = np.array(self.measurements)
		visdom.line(X=X, Y=Y,
			opts = dict(title = 'All Epochs - %s'%(title), legend=self.legend), win=win) 


	def generate_average_XY(self, second_order=False):
		is_average = False
		measure_dim = 1
		for i in range(len(self.measurements)):
			if type(self.measurements[i]) == list:
				is_average = True
				if type(self.measurements[i][0]) == list:
					measure_dim = max(measure_dim, len(self.measurements[i][0]))

		X = np.arange(len(self.measurements))
		Y = np.array(self.measurements)
		skip = False
		if is_average:
			dummy_Y = []
			for i in range(len(Y)):
				if type(Y[i]) == list:
					if self.measure_normalizer is None:
						dummy_Y.append(np.nanmean(np.array(Y[i]), 0))
					else:
						dummy_Y.append(np.nansum(np.array(Y[i]), 0)/self.measure_normalizer)
				if type(Y[i]) == np.ndarray:
					if self.measure_normalizer is None:
						dummy_Y.append(np.nanmean(Y[i], 0))
					else:
						dummy_Y.append(np.nansum(Y[i], 0)/self.measure_normalizer)
				if Y[i] is None:
					if measure_dim > 1:
						dummy_Y.append([float('nan') for q in range(measure_dim)])
					else:
						dummy_Y.append(float('nan'))
					skip = skip or (i==0)
			Y = np.array(dummy_Y)
		else:
			if Y[0] is None:
				skip = True
		if skip:
			X = np.delete(X, 0, axis=0)
			Y = np.delete(Y, 0, axis=0)
		if second_order:
			Y_n = np.zeros(len(Y))
			for i in range(len(Y)):
				Y_n[i] = Y[i].mean()
			Y = Y_n
		return X, Y, is_average

	def visualize_average(self, visdom, second_order=False):
		title = self.measure_name
		win   = self.measure_name
		legend = self.legend
		X, Y, is_average = self.generate_average_XY(second_order)
		if is_average:
			title = 'Average %s'%(title)
			win = 'ave%s'%(win)
			if second_order:
				title = 'Mean %s'%(title)
				win = 'mean%s'%(win)
				legend = None
		visdom.line(X=X, Y=Y,
			opts=dict(title=title, legend=legend), win=win) 

class Logger:
	measures = None

	def __init__(self):
		self.measures = {}

	def log(self, measure_name, measurement, epoch, iteration=None):
		measure = None
		if measure_name in self.measures.keys():
			measure = self.measures[measure_name]
		else:
			measure = Measure(measure_name)
			self.measures[measure_name] = measure

		measure.add_measurement(measurement, epoch, iteration)

	def get_measure(self, measure_name):
		assert measure_name in self.measures.keys(), 'Measure %s is not defined'%measure_name
		return self.measures[measure_name]

	def reset_measure(self, measure_name):
		if measure_name in self.measures.keys():
			self.measures[measure_name].reset()

	def mean_epoch(self, measure_name, epoch=-1):
		measure = self.get_measure(measure_name)
		return measure.mean_epoch(epoch=epoch)

	def sum_epoch(self, measure_name, epoch=-1):
		measure = self.get_measure(measure_name)
		return measure.sum_epoch(epoch=epoch)

	def visualize_epoch(self, measure_name, visdom, epoch=-1):
		measure = self.get_measure(measure_name)
		measure.visualize_epoch(visdom, epoch = epoch)

	def visualize_average(self, measure_name, visdom, second_order=False):
		measure = self.get_measure(measure_name)
		measure.visualize_average(visdom, second_order)

	def visualize_all_average(self, visdom):
		for measure in self.measures.values():
			measure.visualize_average(visdom)

	def visualize_average_keys(self, pattern, title, visdom, is_setup=False, prefix=''):
		pat = re.compile(pattern)
		legend = []
		for key, measure in self.measures.iteritems():
			if pat.match(key):
				nX, nY, _ = measure.generate_average_XY()
				legend.append('%s%s'%(prefix, key))
				if is_setup:
					# visdom.updateTrace(X=nX, Y=nY, win=title, name='%s%s'%(prefix, key))
					visdom.line(X=nX, Y=nY, win=title, name='%s%s'%(prefix, key), update='new')
				else:
					visdom.line(X=nX, Y=nY, win=title)
					is_setup = True
		if len(legend)>0:
			visdom.update_window_opts(win=title, opts=dict(title=title, legend=legend))


	def __str__(self):
		return 'Logger with measures\n(%s)'%(', '.join(self.measures.keys()))

if __name__ == '__main__':
	from visdom import Visdom
	import random
	visdom = Visdom(ipv6=False)
	logger = Logger()

	for epoch in range(0, 10):
		for iteration in range(0, 20):
			logger.log('mIoU', [random.random()+epoch for c in range(5)], epoch, iteration)
			logger.log('train_loss', random.random()+epoch/10, epoch, iteration)
			logger.log('test_loss', random.random()+epoch/10+0.4, epoch, iteration)

	for epoch in range(1, 10):
		for iteration in range(0, 20):
			logger.log('mIoU2', [random.random()+epoch for c in range(4)], epoch, iteration)

	logger.get_measure('mIoU').legend = ['a', 'b', 'c', 'd', 'e']

	logger.visualize_average('mIoU', visdom)
	logger.visualize_average('mIoU', visdom, second_order=True)
	logger.visualize_average('mIoU2', visdom)
	
	logger.visualize_epoch('train_loss', visdom)
	logger.visualize_average('train_loss', visdom)
	logger.visualize_average('test_loss', visdom)
	logger.visualize_average_keys('.*_loss', 'Average Losses', visdom)