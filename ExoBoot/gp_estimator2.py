from tcpip import ServerTCP
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

SERVER_IP = '' # server ip address is 192.168.1.2
RECV_PORT = 8080


def run_gp_estimator(model, exos, ws):
	# Start server and wait for connection
	print('Initializing server.')
	server = ServerTCP('', RECV_PORT)
	server.start_server()
	
	packet_len = exos[0].data.shape[1]+1 # Packet from exo boots contains side integer plus exo data (should be 9 values total).

	inf_flag = np.zeros(2)
	next_side = 0
	t = []
	while True:
		try:
			exo_msg = server.from_client()
			if any(exo_msg):
				exo_msg_list = exo_msg.split('!')[1:]
				exo_msg_arr = np.array([[float(value) for value in msg.split(',')] for msg in exo_msg_list if len(msg.split(','))==packet_len])
				
				exos[0].update(exo_msg_arr[exo_msg_arr[:,0]==0, 1:])
				exos[1].update(exo_msg_arr[exo_msg_arr[:,0]==1, 1:])

				# inf_flag[np.unique(exo_msg_arr[:, 0])] = 1 # Set legs for inference if we got new data
				if any(exo_msg_arr[:,0]==0):
					inf_flag[0] = 1
				if any(exo_msg_arr[:,0]==1):
					inf_flag[1] = 1

			if any(inf_flag):
				if inf_flag[next_side] == 1:
					this_side = next_side
					next_side = abs(next_side-1)
				else:
					this_side = abs(next_side-1)

				model_input = exos[this_side].data.reshape(1, ws, packet_len-1).astype('float32')
				out = model.predict(model_input)
				
				gp = out[0][0, 0]
				ss = out[1][0, 0]

				inf_flag[this_side] = 0
				send_msg = "!" + "{:.0f}".format(this_side) + "," + "{:.5f}".format(gp) + "," + "{:.0f}".format(np.round(ss))
				server.to_client(send_msg)

		except:
			print(traceback.print_exc())
			print('Closing!')
			server.close()
			return 0

def main():
	# Load model
	m_file = ModelRT.choose_model()
	input_shape = ModelRT.get_shape_from_name(m_file)
	if not input_shape:
		input_shape = input('Please enter input shape as d1,d2,d3: ')
		input_shape = tuple([int(i) for i in input_shape.split(',')])
	model = ModelRT(m_file=m_file, input_shape=input_shape)
	model.test_model(num_tests=5, verbose=True)
	ws = input_shape[1]
	num_channels = input_shape[2]

	exo_l = ExoBoot(ws, num_channels)
	exo_r = ExoBoot(ws, num_channels)

	# Run gait phase estimator
	exit_cond = run_gp_estimator(model, (exo_l, exo_r), ws)
	
	return


if __name__=="__main__":
	main()
