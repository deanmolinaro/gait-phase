import time
import socket
import traceback
import numpy as np
from os import listdir, getcwd
import math

import numpy as np
from tensorflow.python.keras.models import Sequential, load_model
import tensorflow as tf
from tensorflow import keras

import os
import gc

gc.enable()
os.environ["CUDA_VISIBLE_DEVICES"] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

SERVER_IP = '' # server ip address is 192.168.0.2
RECV_PORT = 8080


def load_estimator():
	model_files = [f for f in listdir(getcwd()) if '.h5' in f]

	print()
	for i, model_file in enumerate(model_files):
		print(f"[{i}] {model_file}")

	while 1:
		file_select = int(input('Please choose a model file from the menu: '))
		if file_select < len(model_files): break

	model_file = model_files[file_select]

	# Get window size from model file name "_WS<window size>_"
	ws = [int(d[2:]) for d in model_file.split('.')[0].split('_') if 'WS' in d][0]

	# Load model
	model = load_model(getcwd() + '/' + model_file)

	return model, ws


def run_server(recv_conn, model_info):
	# Load model and get input dimensions
	model = model_info[0]
	ws = model_info[1]

	print('Warning - Hard coded input size to 10.')
	input_size = 10

	# Initialize input data for forward pass
	input_data = np.zeros((1, ws, input_size))

	while 1:
		try:
			recv_msg = recv_conn.recv(8192)
			if any(recv_msg):
				start_time = time.time()
				
				# Read and parse any incoming data
				recv_data = np.array([[float(value) for value in msg.split(',')[:-1]] for msg in recv_msg.decode().split('!')[1:] if len(msg.split(',')[:-1])==input_size+1]) # Ignore anything before the first ! (this should just be empty)
				# split = recv_msg.decode().split('!')[-1].split(',') # This only receives last message
				# recv_data = np.reshape(np.array(split, dtype=np.float), (1, len(split))) # This only receives last message
				#print(recv_msg.decode().split('!')[-1].split(',')[:-1])

				recv_data = recv_data[:,:-1].reshape(1, -1, input_size)
				# Delete first n rows in input_data based on how many new instances were received (ideally this is only one instance at a time)
				rows_to_remove = recv_data.shape[1]
				input_data = np.delete(input_data, slice(rows_to_remove), axis=1)

				# Append new instances to end of input_data
				input_data = np.append(input_data, recv_data, axis=1)

				# Make sure input_data size is not too large
				if input_data.shape[1] > ws:
					input_data = input_data[0, -ws:, :].reshape(1, -1, input_size)
					print('Fell behind!')

				# Gait phase estimator forward pass
				start_time = time.time()
				gp_estimate_orig = model.predict(input_data)[-1] # Just use last estimate (there should only be 1 anyways)
				# gp_estimate_orig = model(tf.convert_to_tensor(np.float32(inst_data)))
				# print(time.time()-start_time)

				# # Code for debugging inference time
				# for i in range(100):
				# 	start_time = time.time()
				# 	gp_estimate_orig = model.predict(input_data)
				# 	# gp_estimate_orig = model(tf.convert_to_tensor(np.float32(inst_data)))
				# 	print(time.time()-start_time)

				# Convert gait phase phasor to percentage
				gp_estimate_left = ((np.arctan2(gp_estimate_orig[1], gp_estimate_orig[0]) + (2.0*math.pi)) % (2.0*math.pi)) * 1.0/(2*math.pi)
				gp_estimate_right = ((np.arctan2(gp_estimate_orig[3], gp_estimate_orig[2]) + (2.0*math.pi)) % (2.0*math.pi)) * 1.0/(2*math.pi)
				# print(recv_data)
				# gp_estimate_left = input_data[0, -1, 2]
				# gp_estimate_right = np.max(input_data[0, :, 2])

				# Convert gait phase percentage to message template
				# send_msg = "!" + (",").join(["{:.2f}".format(d) for d in gp_estimate_orig]) # Use this to send raw output of model (depends on output shape of current model)
				send_msg = "!" + "{:.3f}".format(gp_estimate_left) + "," + "{:.3f}".format(gp_estimate_right)

				recv_conn.sendall(send_msg.encode())

		except:
			print(traceback.print_exc())
			print('Closing!')
			recv_conn.close()
			return 0


def start_server():
	recv_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
	recv_socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
	recv_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
	recv_socket.bind((SERVER_IP, RECV_PORT))
	recv_socket.listen(1)
	print('\nWaiting for client to connect.')
	recv_conn, recv_addr = recv_socket.accept()
	recv_socket.close()
	print('Client connected!')
	return recv_conn


def main():
	# Load model
	model_info = load_estimator()
	
	# Start server and wait for connection
	recv_conn = start_server()

	# Run gait phase estimator
	exit_cond = run_server(recv_conn, model_info)
	
	return


if __name__=="__main__":
	main()
