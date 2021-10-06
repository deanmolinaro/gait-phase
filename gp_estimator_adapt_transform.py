import time
import socket
import traceback
import numpy as np
from os import listdir, getcwd, path, getcwd, makedirs
import math
import signal

import numpy as np

import os
import gc

import multiprocessing as mp
# from multiprocessing.sharedctypes import RawArray, Value
from ctypes import c_double

from tensorflow.python.keras.models import Sequential, load_model
import tensorflow as tf
from tensorflow import keras

from scipy.signal import find_peaks, peak_prominences, peak_widths
import pandas as pd

from transforms import Transform

class GaitPhaseEstimator:

	def __init__(self):
		# # Get save data name
		# self.log_file_path = self.start_save_file()

		gc.enable()
		os.environ["CUDA_VISIBLE_DEVICES"] = '-1'
		os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
		self.SERVER_IP = ''
		self.RECV_PORT = 8080
		
		# Start server and wait for connection
		self.recv_conn = 0

		self.q = mp.Queue()
		self.lock = mp.Lock()

		self.update_flag = mp.Event()
		self.update_flag.clear()
		self.adapt_toggle = mp.Event()
		self.adapt_toggle.clear()

		# Transform w/ rotation from mocap
		T = np.array([[0.1935, 0.2979, -0.9348, 0.0270], [0.9811, -0.0637, 0.1828, -0.0027], [-0.0051, -0.9525, -0.3046, -0.1205], [0, 0, 0, 1]])
		# # Transform w/o rotation from mocap
		# T = np.array([[0.0830, 0.0282, -0.9962, 0.0270], [0.9965, 0.0040, 0.0831, -0.0027], [0.0063, -0.9996, -0.0278, -0.1205], [0, 0, 0, 1]])
		self.transform = Transform(T)

	def run(self):
		try:
			processes = []

			self.t = 0

			# toggle_adapt_process = mp.Process(target=self.toggle_adapt)
			# processes.append(toggle_adapt_process)

			run_gp_estimator_process = mp.Process(target=self.run_gp_estimator, args=('TensorFlowModels',))
			processes.append(run_gp_estimator_process)

			run_adapt_process = mp.Process(target=self.run_adapt, args=('TensorFlowModels',))
			processes.append(run_adapt_process)

			[p.start() for p in processes]

			self.toggle_adapt()

			while True:
				time.sleep(10)

		except:
			# Print traceback
			traceback.print_exc()

			# # Terminate all processes
			# pool.terminate()
			# pool.join()

			[p.join() for p in processes]

			if self.run_server:
				# Close TCP connection
				self.recv_conn.close()

			return

	# def choose_estimator(self):
	# 	model_files = [f for f in listdir(getcwd()) if '.h5' in f]

	# 	print()
	# 	for i, model_file in enumerate(model_files):
	# 		print(f"[{i}] {model_file}")

	# 	while 1:
	# 		file_select = int(input('Please choose a model file from the menu: '))
	# 		if file_select < len(model_files): break

	# 	model_file = model_files[file_select]
	# 	return model_file

	# def load_estimator(self):
	# 	# model_files = [f for f in listdir(getcwd()) if '.h5' in f]

	# 	# print()
	# 	# for i, model_file in enumerate(model_files):
	# 	# 	print(f"[{i}] {model_file}")

	# 	# while 1:
	# 	# 	file_select = int(input('Please choose a model file from the menu: '))
	# 	# 	if file_select < len(model_files): break

	# 	# model_file = model_files[file_select]

	# 	# Get window size from model file name "_WS<window size>_"
	# 	ws = [int(d[2:]) for d in self.model_file.split('.')[0].split('_') if 'WS' in d][0]

	# 	# Load model
	# 	model = load_model(getcwd() + '/' + self.model_file)

	# 	return model, ws

	def start_server(self):
		recv_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
		recv_socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
		recv_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
		recv_socket.bind((self.SERVER_IP, self.RECV_PORT))
		recv_socket.listen(1)
		print('\nWaiting for client to connect.')
		recv_conn, recv_addr = recv_socket.accept()
		recv_socket.close()
		print('Client connected!')
		return recv_conn

	def toggle_adapt(self):
		while True:
			if self.adapt_toggle.is_set():
				input('Press enter to end adaptation: ')
				self.adapt_toggle.clear()
			else:
				input('Press enter to start adaptation: ')
				self.adapt_toggle.set()

	def run_adapt(self, ext=''):
		model_dir = getcwd() + '/' + ext
		left_model = load_model(model_dir + '/GP_Left_WS80_noBN.h5')
		right_model = load_model(model_dir + '/GP_Right_WS80_noBN.h5')

		left_adap_model = load_model(model_dir + '/GP_Left_WS80_noBN.h5')
		right_adap_model = load_model(model_dir + '/GP_Right_WS80_noBN.h5')
		opt = keras.optimizers.Adam(learning_rate=0.0001)
		left_adap_model.compile(optimizer=opt, loss='mean_absolute_error')
		right_adap_model.compile(optimizer=opt, loss='mean_absolute_error')

		ws = 80
		input_size = 10
		buf_size = 1000
		buf = np.empty((0, input_size))
		left_buf = np.empty((0, input_size))
		right_buf = np.empty((0, input_size))
		adapting = False

		while True:
			if self.adapt_toggle.is_set():
				while not self.q.empty():
					recv_data = np.squeeze(self.q.get(), axis=0)
					buf = np.append(buf, recv_data, axis=0)

				if buf.shape[0] > buf_size:
					left_peak_idx = self.find_peak_idx(buf[:,0])
					right_peak_idx = self.find_peak_idx(buf[:,1])

					if not left_peak_idx.any() or not right_peak_idx.any():
						print('Warning - Could not find peaks. Clearing buffer.')
						buf = np.empty((0, input_size))
						continue
					else:
						left_peak_idx = left_peak_idx[-1]
						right_peak_idx = right_peak_idx[-1]

					left_x = np.concatenate([left_buf, buf[:left_peak_idx, :]], axis=0)
					right_x = np.concatenate([right_buf, buf[:right_peak_idx, :]], axis=0)

					_, left_end_idx, left_y = self.label_ground_truth(left_x[:,0])
					_, right_end_idx, right_y = self.label_ground_truth(right_x[:,1])

					left_x = left_x[:left_end_idx,:]
					left_y = left_y[:left_end_idx,:]
					right_x = right_x[:right_end_idx,:]
					right_y = right_y[:right_end_idx,:]

					left_x, left_y = self.stride_data(left_x, left_y, ws)
					right_x, right_y = self.stride_data(right_x, right_y, ws)
					left_y = np.squeeze(left_y, axis=1)
					right_y = np.squeeze(right_y, axis=1)

					left_adap_y = left_adap_model.predict(left_x)
					right_adap_y = right_adap_model.predict(right_x)
					left_static_y = left_model.predict(left_x)
					right_static_y = right_model.predict(right_x)

					left_adap_result = self.custom_rmse_uni(left_y, left_adap_y)
					left_static_result = self.custom_rmse_uni(left_y, left_static_y)
					right_adap_result = self.custom_rmse_uni(right_y, right_adap_y)
					right_static_result = self.custom_rmse_uni(right_y, right_static_y)

					# print(str(round(left_adap_result, 3))+" | "+str(round(right_adap_result, 3)))
					print(str(round(left_static_result, 3))+", "+str(round(left_adap_result, 3))+" | "+str(round(right_static_result, 3))+", "+str(round(right_adap_result, 3)))

					left_adap_model.fit(left_x, left_y, epochs=1, verbose=0)
					right_adap_model.fit(right_x, right_y, epochs=1, verbose=0)
					adapting = True

					left_buf = buf[left_peak_idx:, :]
					right_buf = buf[right_peak_idx:, :]
					buf = np.empty((0, input_size))

					self.lock.acquire()
					left_adap_model.save_weights('left_checkpoint')
					right_adap_model.save_weights('right_checkpoint')
					self.lock.release()

					self.update_flag.set()
			elif not self.adapt_toggle.is_set():
				if not self.q.empty():
					print('Clearing adaptation buffer.')
					while not self.q.empty():
						self.q.get()
				if adapting:
					adapting = False
					print('Saving new model.')
					left_adap_model.save(model_dir + '/GP_Left_WS80_noBN_adapted.h5')
					right_adap_model.save(model_dir + '/GP_Right_WS80_noBN_adapted.h5')
					

	def run_gp_estimator(self, ext=''):
		# Load model and get input dimensions
		# model = self.model_info[0]
		# ws = self.model_info[1]

		model_dir = getcwd() + '/' + ext

		# model, ws = self.load_estimator()
		left_model = load_model(model_dir + '/GP_Left_WS80_noBN.h5')
		right_model = load_model(model_dir + '/GP_Right_WS80_noBN.h5')

		self.recv_conn = self.start_server()

		print('Warning - Hard coded input size to 10.')
		input_size = 10

		# Initialize input data for forward pass	
		ws = 80
		input_size = 10
		buffer_size = 1000
		input_data = np.zeros((1, ws, input_size))

		slow_count = 0

		while True:
			try:
				recv_msg = self.recv_conn.recv(8192)
				# recv_msg = '!1,2,3,4,5,6,7,8,9,10,11'
				if any(recv_msg):
					start_time = time.time()

					# Read and parse any incoming data
					# recv_data = np.array([[float(value) for value in msg.split(',')[:-1]] for msg in recv_msg.decode().split('!')[1:] if len(msg.split(',')[:-1])==input_size]).reshape(1, -1, input_size) # Ignore anything before the first ! (this should just be empty)
					recv_data = np.array([[float(value) for value in msg.split(',')[:-1]] for msg in recv_msg.decode().split('!')[1:] if len(msg.split(',')[:-1])==input_size+1]) # Ignore anything before the first ! (this should just be empty)

					timestamp = recv_data[:, -1]
					recv_data = recv_data[:, :-1]

					s_gyro = 0
					e_gyro = 3
					s_accel = 3
					e_accel = 6
					gyro_data = recv_data[:, s_gyro:e_gyro] * (1000/(2**15)) * (np.pi / 180.) # rad/s
					accel_data = recv_data[:, s_accel:e_accel] * (4/(2**15)) * 9.81 # m/s

					gyro_data = self.transform.rotate(gyro_data.transpose())
					accel_data = self.transform.rotate(accel_data.transpose())
					gyro_for_grad = np.concatenate((input_data[0, -1, s_gyro:e_gyro].reshape(-1, 1), gyro_data), axis=1) # Insert last timestep of gyro data at start of array to take derivative
					ang_accel = np.diff(gyro_for_grad, axis=1) / 0.005 # Assuming data is at 200 Hz
					accel_data = self.transform.translate_accel(accel_data, gyro_data, ang_accel)

					recv_data[:, s_gyro:e_gyro] = gyro_data.transpose() * (180. / np.pi) # deg/s
					recv_data[:, s_accel:e_accel] = accel_data.transpose() / 9.81 # G's

					recv_data = recv_data.reshape(1, -1, input_size) # reshape data for conv layers
					# timestamp = recv_data[-1, -1, -1] # Get last timestamp to stamp imu data
					# recv_data = recv_data[:, :, :-1] # Remove timestamp data for inferece

					# if recv_data.shape[1] > 1:
					# 	slow_count += 1
					# 	print('Too slow ' + str(slow_count) + ' ' + str(recv_data.shape[1]))
					# else:
					# 	print('Wooh!')

					# Delete first n rows in input_data based on how many new instances were received (ideally this is only one instance at a time)
					rows_to_remove = recv_data.shape[1]
					input_data = np.delete(input_data, slice(rows_to_remove), axis=1)

					# Append new instances to end of input_data
					input_data = np.append(input_data, recv_data, axis=1)

					# Make sure input_data size is not too large
					if input_data.shape[1] > ws:
						input_data = input_data[0, -ws:, :].reshape(1, -1, input_size)
						print('Fell behind!')

					# Predict current gait phase
					left_gp_estimate_orig = left_model.predict(input_data)[-1]
					right_gp_estimate_orig = right_model.predict(input_data)[-1]
					
					# Convert gait phase phasor to percentage
					gp_estimate_left = ((np.arctan2(left_gp_estimate_orig[1], left_gp_estimate_orig[0]) + (2.0*math.pi)) % (2.0*math.pi)) * 1.0/(2*math.pi)
					gp_estimate_right = ((np.arctan2(right_gp_estimate_orig[1], right_gp_estimate_orig[0]) + (2.0*math.pi)) % (2.0*math.pi)) * 1.0/(2*math.pi)

					# Convert gait phase percentage to message template
					# send_msg = "!" + (",").join(["{:.2f}".format(d) for d in gp_estimate_orig]) # Use this to send raw output of model (depends on output shape of current model)
					send_msg = "!" + "{:.3f}".format(gp_estimate_left) + "," + "{:.3f}".format(gp_estimate_right)
					self.recv_conn.sendall(send_msg.encode())

					if self.adapt_toggle.is_set():
						self.q.put_nowait(recv_data)

				if self.update_flag.is_set():
					# if self.lock.locked():
					# 	print('Cannot acquire lock!')
					# else:
					if self.lock.acquire():
						left_model.load_weights('left_checkpoint')
						right_model.load_weights('right_checkpoint')
						self.lock.release()
					else:
						print('Cannot acquire lock!')

					self.update_flag.clear()

			except:
				print(traceback.print_exc())
				self.recv_conn.close()
				return 0

	def find_peak_idx(self, joint_positions):
	    peaks, _ = find_peaks(joint_positions)
	    prominences = peak_prominences(joint_positions, peaks)[0]
	    maximas, _ = find_peaks(joint_positions, prominence=0.2, distance=100)
	    return maximas

	def label_ground_truth(self, joint_positions):
	    maximas = self.find_peak_idx(joint_positions)
	    maximas = np.append(0, maximas)
	    end_idx = maximas[-1]
	    
	    y = pd.Series(np.nan, index=range(0, joint_positions.shape[0]))  
	    for maxima in maximas:
	        y[maxima] = 1
	        y[maxima+1] = 0
	    y.interpolate(inplace=True)
	    y.fillna(0, inplace=True)
	    y_theta = y * 2 * np.pi
	    
	    cartesian_output = np.stack([np.cos(y_theta), np.sin(y_theta)], axis=1)
	    return y, end_idx, cartesian_output

	def custom_rmse_uni(self, left_true, left_pred):
	    #Raw values and Prediction are in X,Y
	    labels, theta, gp = {}, {}, {}
	    
	    #Calculate cosine distance
	    left_num = np.sum(np.multiply(left_true, left_pred), axis=1)
	    left_denom = np.linalg.norm(left_true, axis=1) * np.linalg.norm(left_pred, axis=1)

	    left_cos = left_num / left_denom
	    
	    #Clip large values and small values
	    left_cos = np.minimum(left_cos, np.zeros(left_cos.shape)+1)
	    left_cos = np.maximum(left_cos, np.zeros(left_cos.shape)-1)
	    
	    # What if denominator is zero (model predicts 0 for both X and Y)
	    left_cos[np.isnan(left_cos)] = 0
	    
	    #Get theta error
	    left_theta = np.arccos(left_cos)
	    
	    #Get gait phase error
	    left_gp_error = left_theta * 100 / (2*np.pi)
	    
	    #Get rmse
	    left_rmse = np.sqrt(np.mean(np.square(left_gp_error)))

	    #Separate legs
	    labels['left_true'] = left_true
	    labels['left_pred'] = left_pred

	    for key, value in labels.items(): 
	        #Convert to polar
	        theta[key] = np.arctan2(value[:, 1], value[:, 0])
	        
	        #Bring into range of 0 to 2pi
	        theta[key] = np.mod(theta[key] + 2*np.pi, 2*np.pi)

	        #Interpolate from 0 to 100%
	        gp[key] = 100*theta[key] / (2*np.pi)

	    return left_rmse

	def stride_data(self, x, y, window_size):
		shape = (x.shape[0] - window_size + 1, window_size, x.shape[1])
		strides = (x.strides[0], x.strides[0], x.strides[1])
		x = np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)
		y = np.expand_dims(y[window_size - 1:], axis=1)
		return x, y

def main():
	gp_estimator = GaitPhaseEstimator()
	gp_estimator.run()


if __name__=="__main__":
	main()