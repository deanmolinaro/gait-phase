import numpy as np


class ExoBoot(object):
	def __init__(self, h, w):
		self.data = np.zeros((h, w))
		self.height = h

	def update(self, new_data):
		if new_data.shape[1] != self.data.shape[1]:
			print('New data wrong size.')
			return -1

		rows_to_remove = new_data.shape[0]
		self.data = np.delete(self.data, slice(rows_to_remove), axis=0)
		self.data = np.append(self.data, new_data, axis=0)
		if self.data.shape[0] > self.height:
			print('Fell behind.')
			self.data = self.data[-self.height:, :]
		return 1