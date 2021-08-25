from ipserver import ServerTCP
from exoboot import ExoBoot
from rtmodels import ModelRT

import time
import traceback
import numpy as np
from os import listdir, getcwd
import os

# from tensorflow.python.keras.models import load_model

import gc

gc.enable()
os.environ["CUDA_VISIBLE_DEVICES"] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

SERVER_IP = '' # server ip address is 192.168.0.2
RECV_PORT = 8080


def load_estimator(ext=''):
	model_dir = getcwd() + '/' + ext
	model_files = [f for f in listdir(model_dir) if '.h5' in f]

	print()
	for i, model_file in enumerate(model_files):
		print(f"[{i}] {model_file}")

	while 1:
		file_select = int(input('Please choose a model file from the menu: '))
		if file_select < len(model_files): break

	model_file = model_files[file_select]

	# Get window size from model file name "_WS<window size>_"
	ws = [int(d[2:]) for d in model_file.split('.')[0].split('_') if 'WS' in d]

	if not any(ws):
		ws = int(input('Please enter window size: '))
	else:
		ws = ws[0]

	# Load model
	model = load_model(model_dir + '/' + model_file)

	return model, ws


def run_gp_estimator(exos, ws):
	# Load model
	# model_info = load_estimator('')
	model = ModelRT(input_shape=(1, ws, 8))
	model.test_model(num_tests=10, verbose=True)

	# Start server and wait for connection
	print('Initializing server.')
	server = ServerTCP('', 8080)
	server.start_server()
	
	packet_len = exos[0].data.shape[1]+1 # Packet from exo boots contains side integer plus exo data (should be 9 values total).

	# pCount = 0 # For printing data
	inf_flag = np.zeros(2)
	next_side = 0
	t = []
	while True:
		try:
			# start_time = time.perf_counter()
			exo_msg = server.from_client()
			if any(exo_msg):
				exo_msg_list = exo_msg.split('!')[1:]
				exo_msg_arr = np.array([[float(value) for value in msg.split(',')] for msg in exo_msg_list if len(msg.split(','))==packet_len])
				# print(exo_msg_arr)
				exos[0].update(exo_msg_arr[exo_msg_arr[:,0]==0, 1:])
				exos[1].update(exo_msg_arr[exo_msg_arr[:,0]==1, 1:])

				# inf_flag[np.unique(exo_msg_arr[:, 0])] = 1 # Set legs for inference if we got new data
				if any(exo_msg_arr[:,0]==0):
					inf_flag[0] = 1
				if any(exo_msg_arr[:,0]==1):
					inf_flag[1] = 1

				# pCount += 1 # For printing data
				# if pCount%10==0: # For printing data
					# print(f'\nLeft Shape: {exos[0].data.shape}') # For printing data
					# print(exos[0].data[-5:, :]) # For printing data
					# print(f'\nRight Shape: {exos[1].data.shape}') # For printing data
					# print(exos[1].data[-5:, :]) # For printing data

				# t.append(time.perf_counter() - start_time)

			if any(inf_flag):
				start_time = time.perf_counter()
				if inf_flag[next_side] == 1:
					this_side = next_side
					next_side = abs(next_side-1)
				else:
					this_side = abs(next_side-1)

				model_input = exos[this_side].data.reshape(1, ws, packet_len-1).astype('float32')
				out = model.predict(model_input)
				# out = model.time_predict(model_input)
				# out = model.run(model_input)
				# dt = time.perf_counter() - start_time
				# print(dt*1000)
				
				# gp = out[0][0][0][0]
				# ss = out[1][0][0][0]
				gp = out[0][0, 0]
				ss = out[1][0, 0]

				inf_flag[this_side] = 0
				send_msg = "!" + "{:.0f}".format(this_side) + "," + "{:.5f}".format(gp) + "," + "{:.0f}".format(np.round(ss))
				server.to_client(send_msg)

				t.append(time.perf_counter() - start_time)

		except:
			print(traceback.print_exc())
			print('Closing!')
			print(t)
			server.close()
			return 0

def main():
	ws = 50
	exo_l = ExoBoot(ws, 8)
	exo_r = ExoBoot(ws, 8)

	# Run gait phase estimator
	exit_cond = run_gp_estimator((exo_l, exo_r), ws)
	
	return


if __name__=="__main__":
	main()
