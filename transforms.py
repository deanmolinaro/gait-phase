import numpy as np 

class Transform():
	def __init__(self, T):
		self.T = T
		self.R = T[:3, :3]
		self.P = T[:3, -1].reshape(-1, 1)

		self.R_inv = self.R.transpose()
		self.T_inv = np.concatenate((self.R_inv, -np.matmul(self.R_inv, self.P)), axis=1)
		self.T_inv = np.concatenate((self.T_inv, np.array([0, 0, 0, 1]).reshape(1, -1)), axis=0)

	def convert_for_transform(self, x):
		if x.shape[0] == 3:
			return np.concatenate((x, np.ones((1, x.shape[1]))), axis=0), True
		return x, False

	def revert_from_transform(self, x):
		x = x[:3, :]
		if x.ndim == 1:
			x = x.reshape(-1, 1)
		return x

	def rotate(self, x):
		return np.matmul(self.R, x)

	def rotate_with_inverse(self, x):
		return np.matmul(self.R_inv, x)

	def safe_transform(func):
		def wrapper(self, x):
			x, converted = self.convert_for_transform(x)
			y = func(self, x)
			if converted:
				y = self.revert_from_transform(y)
			return y
		return wrapper

	@safe_transform
	def transform(self, x):
		return np.matmul(self.T, x)

	@safe_transform
	def transform_with_inverse(self, x):
		return np.matmul(self.T_inv, x)

	def translate_accel(self, accel, ang_vel, ang_accel):
		a_r = []
		w_w_r = []
		for i in range(accel.shape[1]):
			a_r.append(np.cross(ang_accel[:, i].reshape(-1, 1), -self.P, axis=0))
			w_w_r.append(np.cross(ang_vel[:, i].reshape(-1, 1), np.cross(ang_vel[:, i].reshape(-1, 1), -self.P, axis=0), axis=0))
		return accel + np.concatenate(a_r, axis=1) + np.concatenate(w_w_r, axis=1)

if __name__=="__main__":
	T = np.array([[1, 2, 3, 1], [4, 5, 6, 2], [7, 8, 9, 3], [0, 0, 0, 1]])
	my_transform = Transform(T)
	
	# Test vector computation
	print('Testing vector computation... ', end="")
	x = np.array([1, 2, 3]).reshape(-1, 1)
	np.testing.assert_array_equal(my_transform.rotate(x), np.array([14, 32, 50]).reshape(-1, 1))
	np.testing.assert_array_equal(my_transform.rotate_with_inverse(x), np.array([30, 36, 42]).reshape(-1, 1))
	np.testing.assert_array_equal(my_transform.transform(x), np.array([15, 34, 53]).reshape(-1, 1))
	np.testing.assert_array_equal(my_transform.transform_with_inverse(x), np.zeros((3, 1)))
	print('Passed.')

	# Test matrix computation
	print('Testing matrix computation... ', end="")
	x = np.array([[1, 2, 3], [3, 2, 1], [-1, -2, -3], [0.3, -4, -0.1]]).transpose()
	np.testing.assert_allclose(my_transform.rotate(x), np.array([[14, 10, -14, -8], [32, 28, -32, -19.4], [50, 46, -50, -30.8]]))
	np.testing.assert_allclose(my_transform.rotate_with_inverse(x), np.array([[30, 18, -30, -16.4], [36, 24, -36, -20.2], [42, 30, -42, -24]]))
	np.testing.assert_allclose(my_transform.transform(x), np.array([[15, 11, -13, -7], [34, 30, -30, -17.4], [53, 49, -47, -27.8]]))
	np.testing.assert_allclose(my_transform.transform_with_inverse(x), np.array([[0, -12, -60, -46.4], [0, -12, -72, -56.2], [0, -12, -84, -66]]))
	print('Passed.')

	# # Test accelerometer translation
	# print('Testing accelerometer translation... ', end="")
	# accel = np.random.rand(3, 100)
	# ang_vel = np.random.rand(3, 100)
	# ang_accel = np.diff(ang_vel, axis=1)
	# ang_accel = np.concatenate((np.zeros((3, 1)), ang_accel), axis=1)
	# y = my_transform.translate_accel(accel, ang_vel, ang_accel)
	# print('Passed.')

	# Test IMU transformation using preshifted data from virtual IMUs (rotation and translation in data described by transformation matrix)
	print('Testing IMU transformation.')
	import pandas as pd
	R = np.array([[0, -0.2588, 0.9659], [0, 0.9659, 0.2588], [-1, 0, 0]])
	P = np.array([-0.0761, 0.2060, -0.0019]).reshape(-1, 1)
	P = np.matmul(R, P)
	T = np.concatenate((np.concatenate((R, P), axis=1), np.array([0, 0, 0, 1]).reshape(1, -1)), axis=0)
	imu_transform = Transform(T)

	imu_orig = pd.read_csv('./data/vimu_orig.csv')
	accel_orig = imu_orig.loc[:, ('accel_x', 'accel_y', 'accel_z')].values*9.81
	gyro_orig = imu_orig.loc[:, ('gyro_x', 'gyro_y', 'gyro_z')].values

	imu_shift = pd.read_csv('./data/vimu_shift.csv')
	accel_shift = imu_shift.loc[:, ('accel_x', 'accel_y', 'accel_z')].values*9.81
	gyro_shift = imu_shift.loc[:, ('gyro_x', 'gyro_y', 'gyro_z')].values

	gyro_orig_rot = imu_transform.rotate(gyro_orig.transpose()).transpose()
	err = abs(gyro_orig_rot - gyro_shift).max()
	if err > 1e-4:
		raise Exception(f"Transformed gyro does not match! Error = {err}.")

	ang_accel_orig_rot = np.concatenate((np.zeros((1, 3)), np.diff(gyro_orig_rot, axis=0)/0.005), axis=0).transpose()
	accel_orig_rot = imu_transform.rotate(accel_orig.transpose())
	accel_orig_rot_trans = imu_transform.translate_accel(accel_orig_rot, gyro_orig_rot.transpose(), ang_accel_orig_rot).transpose()
	err = abs(accel_orig_rot_trans - accel_shift).max()
	if err > 1.5:
		raise Exception(f"Transformed accel does not match! Error = {err}.")

	# from matplotlib import pyplot as plt
	# plt.plot(gyro_orig_rot)
	# plt.plot(gyro_shift)
	# plt.show()

	# plt.plot(accel_shift)
	# plt.plot(accel_orig_rot_trans)
	# plt.show()

	print('Passed.')