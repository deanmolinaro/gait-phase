import time
import socket
import traceback
import numpy as np
from os import listdir, getcwd
import math
from tensorflow.python.keras.models import Sequential, load_model
import tensorflow as tf
from tensorflow import keras
import os
import gc
import pandas as pd

from scipy.signal import find_peaks, peak_prominences, peak_widths


gc.enable()
os.environ["CUDA_VISIBLE_DEVICES"] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

SERVER_IP = '' # server ip address is 192.168.0.2
RECV_PORT = 8080


def find_peak_idx(joint_positions):
    peaks, _ = find_peaks(joint_positions)
    prominences = peak_prominences(joint_positions, peaks)[0]
    maximas, _ = find_peaks(joint_positions, prominence=0.2, distance=100)
    return maximas

def label_ground_truth(joint_positions):
    maximas = find_peak_idx(joint_positions)
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

def load_estimator():
	model_files = [f for f in listdir(getcwd() + '/../TensorFlowModels') if '.h5' in f]
	print()
	for i, model_file in enumerate(model_files):
		print(f"[{i}] {model_file}")

	while 1:
		file_select = int(input('Please choose a left model file from the menu: '))
		if file_select < len(model_files): break

	model_file = model_files[file_select]

	# Get window size from model file name "_WS<window size>_"
	ws = [int(d[2:]) for d in model_file.split('.')[0].split('_') if 'WS' in d][0]

	# Load model
	left_model = load_model(getcwd() + '/' + model_file)
	left_adap_model = load_model(getcwd() + '/' + model_file)

	model_files = [f for f in listdir(getcwd()) if '.h5' in f]
	print()
	for i, model_file in enumerate(model_files):
		print(f"[{i}] {model_file}")

	while 1:
		file_select = int(input('Please choose a right model file from the menu: '))
		if file_select < len(model_files): break

	model_file = model_files[file_select]

	# Load model
	right_model = load_model(getcwd() + '/' + model_file)
	right_adap_model = load_model(getcwd() + '/' + model_file)

	return left_model, left_adap_model, right_model, right_adap_model, ws

def stride_data(x, y, window_size):
	shape = (x.shape[0] - window_size + 1, window_size, x.shape[1])
	strides = (x.strides[0], x.strides[0], x.strides[1])
	x = np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)
	y = np.expand_dims(y[window_size - 1:], axis=1)
	return x, y

def custom_rmse_uni(left_true, left_pred):
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

def run_server(recv_conn, ext=''):
	# Load model and get input dimensions'
	#left_model = model_info[0]
	#left_adap_model = model_info[1]
	#right_model = model_info[2]
	#right_adap_model = model_info[3]
	#ws = model_info[4]

	model_dir = getcwd() + '/' + ext

	left_model = load_model(model_dir + '/GP_Left_WS80_noBN.h5')
	left_adap_model = load_model(model_dir + '/GP_Left_WS80_noBN.h5')
	right_model = load_model(model_dir + '/GP_Right_WS80_noBN.h5')
	right_adap_model = load_model(model_dir + '/GP_Right_WS80_noBN.h5')

	# Initialize input data for forward pass	
	ws = 80
	input_size = 10
	buffer_size = 1000
	input_data = np.zeros((1, ws, input_size))
	left_old_buffer = np.empty((0, 10))
	right_old_buffer = np.empty((0, 10))
	old_buffer = np.empty((0, 10))
	current_buffer = np.empty((0, 10))

	while 1:
		try:
			recv_msg = recv_conn.recv(8192)
			if any(recv_msg):
				
				# Read and parse any incoming data
				recv_data = np.array([[float(value) for value in msg.split(',')[:-1]] for msg in recv_msg.decode().split('!')[1:] if len(msg.split(',')[:-1])==input_size+1])
				recv_data = recv_data.reshape(1, -1, input_size+1)
				recv_data = recv_data[:, :, :-1]

				rows_to_remove = recv_data.shape[1]
				input_data = np.delete(input_data, slice(rows_to_remove), axis=1)
				input_data = np.append(input_data, recv_data, axis=1)

				# Make sure input_data size is not too large
				if input_data.shape[1] > ws:
					input_data = input_data[0, -ws:, :].reshape(1, -1, input_size)
					# print('Fell behind!')

				# Predict current gait phase
				left_gp_estimate_orig = left_model.predict(input_data)[-1]
				right_gp_estimate_orig = right_model.predict(input_data)[-1]
				
				# Convert gait phase phasor to percentage
				gp_estimate_left = ((np.arctan2(left_gp_estimate_orig[1], left_gp_estimate_orig[0]) + (2.0*math.pi)) % (2.0*math.pi)) * 1.0/(2*math.pi)
				gp_estimate_right = ((np.arctan2(right_gp_estimate_orig[1], right_gp_estimate_orig[0]) + (2.0*math.pi)) % (2.0*math.pi)) * 1.0/(2*math.pi)

    				# Append streaming data for adapting data buffer
				current_buffer = np.append(current_buffer, np.squeeze(recv_data, axis = 0), axis=0)

				# Adaptation Code
				if current_buffer.shape[0] > buffer_size:

					left_peak_idx = find_peak_idx(current_buffer[:,0])[-1]
					right_peak_idx = find_peak_idx(current_buffer[:,1])[-1]

					left_x = np.concatenate([left_old_buffer, current_buffer[:left_peak_idx, :]], axis=0)
					right_x = np.concatenate([right_old_buffer, current_buffer[:right_peak_idx, :]], axis=0)

					_, left_end_idx, left_y = label_ground_truth(left_x[:,0])
					_, right_end_idx, right_y = label_ground_truth(right_x[:,0])

					left_x = left_x[:left_end_idx,:]
					left_y = left_y[:left_end_idx,:]
					right_x = right_x[:right_end_idx,:]
					right_y = right_y[:right_end_idx,:]

					left_x, left_y = stride_data(left_x, left_y, ws)
					right_x, right_y = stride_data(right_x, right_y, ws)
					left_y = np.squeeze(left_y, axis=1)
					right_y = np.squeeze(right_y, axis=1)

					left_adap_y = left_adap_model.predict(left_x)
					right_adap_y = right_adap_model.predict(right_x)
					left_static_y = left_model.predict(left_x)
					right_static_y = right_model.predict(right_x)

					left_adap_result = custom_rmse_uni(left_y, left_adap_y)
					left_static_result = custom_rmse_uni(left_y, left_static_y)
					right_adap_result = custom_rmse_uni(right_y, right_adap_y)
					right_static_result = custom_rmse_uni(right_y, right_static_y)

					print(str(round(left_static_result, 3))+", "+str(round(left_adap_result, 3))+" | "+str(round(right_static_result, 3))+", "+str(round(right_adap_result, 3)))

					left_adap_model.fit(left_x, left_y, epochs=1, verbose=0)
					right_adap_model.fit(right_x, right_y, epochs=1, verbose=0)

					left_old_buffer = current_buffer[left_peak_idx:, :]
					right_old_buffer = current_buffer[right_peak_idx:, :]
					current_buffer = np.empty((0, 10))

				# Convert gait phase percentage to message template
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
	# model_info = load_estimator()

	# Start server and wait for connection
	recv_conn = start_server()

	# Run gait phase estimator
	exit_cond = run_server(recv_conn, 'TensorFlowModels')
	
	return


if __name__=="__main__":
	main()
