import time
import socket
import traceback
import numpy as np
from os import listdir, getcwd, path, getcwd, makedirs
import math
import signal

import os
import gc

import multiprocessing as mp
from multiprocessing.sharedctypes import RawArray, Value
from ctypes import c_double

from imu.MPU9250 import MPU9250
import board
import digitalio

from tensorflow.python.keras.models import Sequential, load_model
import tensorflow as tf
from tensorflow import keras

class GaitPhaseEstimator:

	def __init__(self, run_server=False):
		# Get save data name
		self.log_file_path = self.start_save_file()

		gc.enable()
		os.environ["CUDA_VISIBLE_DEVICES"] = '-1'
		os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
		self.SERVER_IP = ''
		self.RECV_PORT = 8080

		# Set up sync
		self.sync_pin = digitalio.DigitalInOut(board.D18) # Was D4
		self.sync_pin.direction = digitalio.Direction.INPUT
		self.sync_pin.pull = digitalio.Pull.DOWN
		self.syncPrev = 0

		# self.test_pin = digitalio.DigitalInOut(board.D18)
		# self.test_pin.direction = digitalio.Direction.OUTPUT 
		# self.test_pin.value = 1

		# while True:
		# 	print(int(self.sync_pin.value))
		# 	time.sleep(0.1)

		# Set up IMUs
		self.SCL_l = board.SCL
		self.SDA_l = board.SDA
		self.SCL_r = board.SCL_1
		self.SDA_r = board.SDA_1
		self.imu_l = self.start_imu(scl=self.SCL_l, sda=self.SDA_l)
		self.imu_r = self.start_imu(scl=self.SCL_r, sda=self.SDA_r)
		
		# Start server and wait for connection
		self.run_server = run_server
		if self.run_server:
			self.recv_conn = 0
			# self.model_info = self.load_estimator() # Load model
			# self.recv_conn = self.start_server()
			self.model_file = self.choose_estimator('TensorFlowModels')

		# self.imu_data = np.zeros((1, 14))
		self.imu_data = RawArray(c_double, 1000*15)
		self.imu_data_idx = Value('i', -15)
		self.lock = mp.Lock()

	def run(self):
		try:
			processes = []
			imu_process = mp.Process(target=self.log_imu_data, args=(0.005,))
			processes.append(imu_process)

			# Test that mp.Lock() works
			# slow_process = mp.Process(target=self.slow)
			# processes.append(slow_process)

			save_data_process = mp.Process(target=self.save_data)
			processes.append(save_data_process)

			if self.run_server:
				run_gp_estimator_process = mp.Process(target=self.run_gp_estimator)
				processes.append(run_gp_estimator_process)

			[p.start() for p in processes]

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

			print('Exiting!')

			return

	def init_worker(self):
		# Ignore normal keyboard interrupt exit to properly close multiprocessing
		signal.signal(signal.SIGINT, signal.SIG_IGN)

	def choose_estimator(self, ext=''):
		model_dir = getcwd() + '/' + ext
		model_files = [f for f in listdir(model_dir) if '.h5' in f]

		print()
		for i, model_file in enumerate(model_files):
			print(f"[{i}] {model_file}")

		while 1:
			file_select = int(input('Please choose a model file from the menu: '))
			if file_select < len(model_files): break

		model_file = model_dir + '/' + model_files[file_select]
		return model_file

	def load_estimator(self):
		# model_files = [f for f in listdir(getcwd()) if '.h5' in f]

		# print()
		# for i, model_file in enumerate(model_files):
		# 	print(f"[{i}] {model_file}")

		# while 1:
		# 	file_select = int(input('Please choose a model file from the menu: '))
		# 	if file_select < len(model_files): break

		# model_file = model_files[file_select]

		# Get window size from model file name "_WS<window size>_"
		ws = [int(d[2:]) for d in self.model_file.split('.')[0].split('_') if 'WS' in d][0]

		# Load model
		model = load_model(self.model_file)

		return model, ws

	def log_imu_data(self, loop_time=0.005):
		try:
			print('Starting imu logging.')
			begin_time = time.time()
			start_time = begin_time
			count = 0
			while True:
				if time.time()-start_time >= loop_time:
						start_time = time.time()
						try:
							imu_data_l = self.imu_l.get_imu_data() # TODO: Using threading here
						except KeyboardInterrupt:
							raise KeyboardInterrupt
						except:
							print('Rebooting left IMU')
							while True:
								try:
									self.imu_l = self.start_imu(scl=self.SCL_l, sda=self.SDA_l)
									print('Down for ' + str(time.time()-start_time) + 's')
									break
								except KeyboardInterrupt:
									raise KeyboardInterrupt
								except:
									pass
							continue
						try:
							imu_data_r = self.imu_r.get_imu_data() # TODO: Using threading here
						except KeyboardInterrupt:
							raise KeyboardInterrupt
						except:
							print('Rebooting right IMU')
							while True:
								try:
									self.imu_r = self.start_imu(scl=self.SCL_r, sda=self.SDA_r)
									print('Down for ' + str(time.time()-start_time) + 's')
									break
								except KeyboardInterrupt:
									raise KeyboardInterrupt
								except:
									pass
							continue
							
						sync = int(self.sync_pin.value)
						if sync != self.syncPrev:
							print('Last Sync = ' + str(sync))
							self.syncPrev = sync
						self.update_imu_data(start_time-begin_time, imu_data_l, imu_data_r, sync)
						# print(f"{imu_data_l[0]}, {imu_data_r[0]}")
		except:
			print('Imu logging failed.')
			traceback.print_exc()

	def update_imu_data(self, imu_time, imu_data_l, imu_data_r, sync):
		self.lock.acquire()
		# self.imu_data[0, :] = (imu_time,)+imu_data_l+imu_data_r+(0.,)
		self.imu_data_idx.value += 15
		idx = self.imu_data_idx.value
		self.imu_data[idx] = imu_time
		self.imu_data[idx+1] = imu_data_l[0]
		self.imu_data[idx+2] = imu_data_l[1]
		self.imu_data[idx+3] = imu_data_l[2]
		self.imu_data[idx+4] = imu_data_l[3]
		self.imu_data[idx+5] = imu_data_l[4]
		self.imu_data[idx+6] = imu_data_l[5]
		self.imu_data[idx+7] = imu_data_r[0]
		self.imu_data[idx+8] = imu_data_r[1]
		self.imu_data[idx+9] = imu_data_r[2]
		self.imu_data[idx+10] = imu_data_r[3]
		self.imu_data[idx+11] = imu_data_r[4]
		self.imu_data[idx+12] = imu_data_r[5]
		self.imu_data[idx+13] = sync
		self.imu_data[idx+14] = 0.
		# self.imu_data_idx.value += 15

		# self.imu_data = np.append(self.imu_data, np.array((imu_time,)+imu_data_l+imu_data_r+(0.,)).reshape(1, -1), axis=0)
		# print('Update ', end=" ")
		# print(self.imu_data_idx.value)
		self.lock.release()
		return 1

	def save_data(self):
		try:
			while True:
				imu_save_data = self.get_imu_save_data()

				if imu_save_data.any():
					# print("Got save data!", end=" ")
					# print(imu_save_data.shape)

					with open(self.log_file_path, 'a') as f:
						np.savetxt(f, imu_save_data, delimiter=",")

				time.sleep(0.1)
		except:
			traceback.print_exc()
			return 0

	def get_imu_save_data(self):
		self.lock.acquire()
		idx = self.imu_data_idx.value

		if idx > 0: # Leave the last value in the queue in case timestamp comes from exo
			imu_save_data = np.asarray(self.imu_data[:idx-15]).reshape(-1, 15)
			self.imu_data[:15] = self.imu_data[idx-15:idx]
			self.imu_data[15:idx] = [0.]*(idx-15)
			self.imu_data_idx.value = 0
		else:
			imu_save_data = np.array(0.)

		self.lock.release()
		return imu_save_data

	# def slow(self):
	# 	while True:
	# 		self.lock.acquire()
	# 		time.sleep(2)
	# 		self.lock.release()
	# 		time.sleep(1)

	def start_imu(self, scl=board.SCL, sda=board.SDA):
		# Initialize IMUs
		imu = MPU9250(scl=scl, sda=sda)
		time.sleep(0.1)

		if scl==board.SCL and sda==board.SDA:
			# imu.set_accel_offset_x(-3120)
			# imu.set_accel_offset_y(2902)
			# imu.set_accel_offset_z(4096)
			# imu.set_gyro_offset_x(573)
			# imu.set_gyro_offset_y(-78)
			# imu.set_gyro_offset_z(-25)

			# imu.set_accel_offset_x(-3800) # IMU1 settings (updated before start of pilots)
			# imu.set_accel_offset_y(3500)
			# imu.set_accel_offset_z(3650)
			# imu.set_gyro_offset_x(-20)
			# imu.set_gyro_offset_y(50)
			# imu.set_gyro_offset_z(-10)

			imu.set_accel_offset_x(-1550) # IMU3 settings (Updated before start of pilots)
			imu.set_accel_offset_y(-2150)
			imu.set_accel_offset_z(5000)
			imu.set_gyro_offset_x(30)
			imu.set_gyro_offset_y(5)
			imu.set_gyro_offset_z(-50)

		elif scl==board.SCL_1 and sda==board.SDA_1:
			# imu.set_accel_offset_x(-3150) # IMU2 settings (old, before collection of AB02)
			# imu.set_accel_offset_y(2900)
			# imu.set_accel_offset_z(4100)
			# imu.set_gyro_offset_x(575)
			# imu.set_gyro_offset_y(-75)
			# imu.set_gyro_offset_z(-20)
			imu.set_accel_offset_x(-3817) # IMU2 settings (Updated 12102020, after collection of AB02)
			imu.set_accel_offset_y(3567)
			imu.set_accel_offset_z(3625)
			imu.set_gyro_offset_x(-20)
			imu.set_gyro_offset_y(50)
			imu.set_gyro_offset_z(-20)

		imu.set_accel_scale(4)
		imu.set_gyro_scale(1000)
		return imu

	def start_save_file(self):
		if not path.exists(getcwd() + "/log"):
			print("Creating " + getcwd() + "/log")
			makedirs("log")

		log_file_names = listdir("log")
		print("Current log file names: ")
		[print(n) for n in log_file_names]

		log_file_path = getcwd() + "/log/" + input("Enter log file name for this trial: ") + ".txt"
		print("Saving data to " + log_file_path + "\n")

		with open(log_file_path, 'w') as f:
			f.write("imuTime,lAccX,lAccY,lAccZ,lGyroX,lGyroY,lGyroZ,rAccX,rAccY,rAccZ,rGyroX,rGyroY,rGyroZ,sync,exoTime\n")

		return log_file_path

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

	def run_gp_estimator(self):
		# Load model and get input dimensions
		# model = self.model_info[0]
		# ws = self.model_info[1]

		model, ws = self.load_estimator()
		self.recv_conn = self.start_server()

		print('Warning - Hard coded input size to 10.')
		input_size = 10

		# Initialize input data for forward pass
		input_data = np.zeros((1, ws, input_size))

		slow_count = 0

		while True:
			try:
				recv_msg = self.recv_conn.recv(8192)
				if any(recv_msg):
					start_time = time.time()

					# Read and parse any incoming data
					# recv_data = np.array([[float(value) for value in msg.split(',')[:-1]] for msg in recv_msg.decode().split('!')[1:] if len(msg.split(',')[:-1])==input_size]).reshape(1, -1, input_size) # Ignore anything before the first ! (this should just be empty)
					recv_data = np.array([[float(value) for value in msg.split(',')[:-1]] for msg in recv_msg.decode().split('!')[1:] if len(msg.split(',')[:-1])==input_size+1]) # Ignore anything before the first ! (this should just be empty)

					recv_data = recv_data.reshape(1, -1, input_size+1) # reshape data for conv layers
					timestamp = recv_data[-1, -1, -1] # Get last timestamp to stamp imu data
					recv_data = recv_data[:, :, :-1] # Remove timestamp data for inferece

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

					gp_estimate_orig = model.predict(input_data)[-1] # Just use last estimate (there should only be 1 anyways)

					# Convert gait phase phasor to percentage
					gp_estimate_left = ((np.arctan2(gp_estimate_orig[1], gp_estimate_orig[0]) + (2.0*math.pi)) % (2.0*math.pi)) * 1.0/(2*math.pi)
					gp_estimate_right = ((np.arctan2(gp_estimate_orig[3], gp_estimate_orig[2]) + (2.0*math.pi)) % (2.0*math.pi)) * 1.0/(2*math.pi)

					# Convert gait phase percentage to message template
					# send_msg = "!" + (",").join(["{:.2f}".format(d) for d in gp_estimate_orig]) # Use this to send raw output of model (depends on output shape of current model)
					send_msg = "!" + "{:.3f}".format(gp_estimate_left) + "," + "{:.3f}".format(gp_estimate_right)
					self.recv_conn.sendall(send_msg.encode())

					# Add exo timestamp here. but first need to update exo out and this read to handle for the timestamp
					if self.imu_data_idx.value >= 0:
						self.lock.acquire()
						self.imu_data[self.imu_data_idx.value+14] = timestamp
						self.lock.release()
			except:
				print(traceback.print_exc())
				self.recv_conn.close()
				return 0

def main():
	gp_estimator = GaitPhaseEstimator(run_server=True)
	gp_estimator.run()


if __name__=="__main__":
	main()