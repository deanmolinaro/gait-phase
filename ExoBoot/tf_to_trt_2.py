from tensorflow.python.keras.models import load_model
import argparse
import subprocess

import keras2onnx
import onnx
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import time
from rtmodels import ModelRT
from os import listdir, getcwd


def from_menu(m_dir=None):
	file_names = [f for f in listdir(m_dir) if '.h5' in f]

	print()
	for i, file_name in enumerate(file_names):
		print(f"[{i}] {file_name}")

	while 1:
		file_select = int(input('Please select model to convert: '))
		if file_select < len(file_names): break

	m_file = m_dir + '/' + file_names[file_select] if m_dir else file_names[file_select]

	return m_file

def to_trt(m='', m_dir='', b=None, ws=None, c=None):
	m_dir = m_dir if any(m_dir) else getcwd()
	if not any(m):
		m = from_menu(m_dir)
	m_onnx = m.replace('.h5', '.onnx')
	m_trt = m.replace('.h5', '.trt')

	if not b:
		b = int(input('Please input batch size: '))

	if not ws:
		ws = int(input('Please input window size: '))

	if not c:
		c = int(input('Please input number of input channels: '))

	print(f'Loading {m}...')
	model = load_model(m)

	onnx_model = keras2onnx.convert_keras(model, model.name)

	inputs = onnx_model.graph.input
	for i in inputs:
		dim1 = i.type.tensor_type.shape.dim[0]
		dim1.dim_value = b

	print(f'Converting {m} to {m_onnx}...')
	onnx.save_model(onnx_model, m_onnx)
	print('Done.')

	print(f'Converting {m_onnx} to {m_trt}...')
	cmd = ['trtexec', '--onnx=' + m_onnx, '--saveEngine=' + m_trt]
	subprocess.call(cmd)
	print('Done.')

	print(f'Loading {m_trt}...')
	model = ModelRT(m_trt, input_shape=(1, ws, c))
	print('Done.')

	print(f'Testing {m_trt}...')
	out, t = model.test_model(num_tests=10, verbose=True)
	print('Done.')

	return


if __name__=="__main__":
	# Input argument parser
	parser = argparse.ArgumentParser()
	parser.add_argument('-m', type=str, default='')
	parser.add_argument('-b', type=int, default=None)
	parser.add_argument('-ws', type=int, default=None)
	parser.add_argument('-c', type=int, default=None)

	# Parse input argument
	args = parser.parse_args()
	args_attributes = [att for att in dir(args) if '__' not in att and '_' != att[0]]
	args_dict = {att: getattr(args, att) for att in args_attributes}

	# Convert model file to .trt
	to_trt(**args_dict)
