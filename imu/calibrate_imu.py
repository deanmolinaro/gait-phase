from MPU9250 import MPU9250
import board
import numpy as np
import time

NUM_SAMPLES = 100

imu = MPU9250(sda=board.SDA, scl=board.SCL)

# print(imu.get_accel_offset_x())
# print(imu.get_accel_offset_y())
# print(imu.get_accel_offset_z())

# print(imu.get_gyro_offset_x())
# print(imu.get_gyro_offset_y())
# print(imu.get_gyro_offset_z())
# exit()

print("Setting accel & gyro scale & offset for calibration.")
imu.set_accel_scale(16)
# imu.set_accel_scale(4)
imu.set_gyro_scale(1000)

imu.set_accel_offset_x(0)
imu.set_accel_offset_y(0)
imu.set_accel_offset_z(0)
imu.set_gyro_offset_x(0)
imu.set_gyro_offset_y(0)
imu.set_gyro_offset_z(0)
print("Done.\n")

print("Starting accelerometer calibration.")
print("Assuming z-axis is pointing up.")
accel_x = []
accel_y = []
accel_z = []

print("Sampling accelerometer data.")
for i in range(NUM_SAMPLES):
	d = imu.get_accel_data(keep_int=True)
	accel_x.append(d[0])
	accel_y.append(d[1])
	accel_z.append(d[2])
	time.sleep(0.02)
print("Done.\n")

accel_x = np.array(accel_x)
accel_y = np.array(accel_y)
accel_z = np.array(accel_z)

accel_x_avg = np.mean(accel_x)
accel_y_avg = np.mean(accel_y)
accel_z_avg = np.mean(accel_z)

accel_x_std = np.std(accel_x)
accel_y_std = np.std(accel_y)
accel_z_std = np.std(accel_z)

print("Uncalibrated accelerometer results:")
print(f"Accel X = {accel_x_avg} +/- {accel_x_std}")
print(f"Accel Y = {accel_y_avg} +/- {accel_y_std}")
print(f"Accel Z = {accel_z_avg} +/- {accel_z_std}")
print()

print("Updating accelerometer offset.")
print("Still assuming z-axis is pointing up.")
accel_x_offset = int(-accel_x_avg/2)
accel_y_offset = int(-accel_y_avg/2)
accel_z_offset = int((-1*accel_z_avg + 2048)/2)

imu.set_accel_offset_x(accel_x_offset)
imu.set_accel_offset_y(accel_y_offset)
imu.set_accel_offset_z(accel_z_offset)
print("Done.\n")

print("Sampling accelerometer data with updated offsets.")
accel_x = []
accel_y = []
accel_z = []
for i in range(NUM_SAMPLES):
	d = imu.get_accel_data(keep_int=True)
	accel_x.append(d[0])
	accel_y.append(d[1])
	accel_z.append(d[2])
	time.sleep(0.02)
print("Done.\n")

accel_x = np.array(accel_x)
accel_y = np.array(accel_y)
accel_z = np.array(accel_z)

accel_x_avg = np.mean(accel_x)
accel_y_avg = np.mean(accel_y)
accel_z_avg = np.mean(accel_z)

accel_x_std = np.std(accel_x)
accel_y_std = np.std(accel_y)
accel_z_std = np.std(accel_z)

print("Calibrated accelerometer results:")
print(f"Accel X = {accel_x_avg} +/- {accel_x_std}")
print(f"Accel Y = {accel_y_avg} +/- {accel_y_std}")
print(f"Accel Z = {accel_z_avg} +/- {accel_z_std}")
print()

print("Starting gyroscope calibration.")
gyro_x = []
gyro_y = []
gyro_z = []

print("Sampling gyroscope data.")
for i in range(NUM_SAMPLES):
	d = imu.get_gyro_data(keep_int=True)
	gyro_x.append(d[0])
	gyro_y.append(d[1])
	gyro_z.append(d[2])
	time.sleep(0.02)
print("Done.\n")

gyro_x = np.array(gyro_x)
gyro_y = np.array(gyro_y)
gyro_z = np.array(gyro_z)

gyro_x_avg = np.mean(gyro_x)
gyro_y_avg = np.mean(gyro_y)
gyro_z_avg = np.mean(gyro_z)

gyro_x_std = np.std(gyro_x)
gyro_y_std = np.std(gyro_y)
gyro_z_std = np.std(gyro_z)

print("Uncalibrated gyroscope results:")
print(f"Gyro X = {gyro_x_avg} +/- {gyro_x_std}")
print(f"Gyro Y = {gyro_y_avg} +/- {gyro_y_std}")
print(f"Gyro Z = {gyro_z_avg} +/- {gyro_z_std}")
print()

print("Updating gyroscope offset.")
gyro_x_offset = int(-gyro_x_avg)
gyro_y_offset = int(-gyro_y_avg)
gyro_z_offset = int(-gyro_z_avg)

imu.set_gyro_offset_x(gyro_x_offset)
imu.set_gyro_offset_y(gyro_y_offset)
imu.set_gyro_offset_z(gyro_z_offset)
print("Done.\n")

print("Sampling gyroscope data with updated offset.")
gyro_x = []
gyro_y = []
gyro_z = []
for i in range(NUM_SAMPLES):
	d = imu.get_gyro_data(keep_int=True)
	gyro_x.append(d[0])
	gyro_y.append(d[1])
	gyro_z.append(d[2])
	time.sleep(0.02)
print("Done.\n")

gyro_x = np.array(gyro_x)
gyro_y = np.array(gyro_y)
gyro_z = np.array(gyro_z)

gyro_x_avg = np.mean(gyro_x)
gyro_y_avg = np.mean(gyro_y)
gyro_z_avg = np.mean(gyro_z)

gyro_x_std = np.std(gyro_x)
gyro_y_std = np.std(gyro_y)
gyro_z_std = np.std(gyro_z)

print("Calibrated gyroscope results:")
print(f"Gyro X = {gyro_x_avg} +/- {gyro_x_std}")
print(f"Gyro Y = {gyro_y_avg} +/- {gyro_y_std}")
print(f"Gyro Z = {gyro_z_avg} +/- {gyro_z_std}")
print()

print("Changing accelerometer & gyroscope scales for final test.")
imu.set_accel_scale(4)
imu.set_gyro_scale(1000)
print("Done.\n")

print("Sampling accelerometer & gyroscope data.")
accel_x = []
accel_y = []
accel_z = []
gyro_x = []
gyro_y = []
gyro_z = []
for i in range(NUM_SAMPLES):
	a_d = imu.get_accel_data(keep_int=False)
	accel_x.append(a_d[0])
	accel_y.append(a_d[1])
	accel_z.append(a_d[2])

	g_d = imu.get_gyro_data(keep_int=False)
	gyro_x.append(g_d[0])
	gyro_y.append(g_d[1])
	gyro_z.append(g_d[2])

	time.sleep(0.02)
print("Done.\n")

accel_x = np.array(accel_x)
accel_y = np.array(accel_y)
accel_z = np.array(accel_z)

accel_x_avg = np.mean(accel_x)
accel_y_avg = np.mean(accel_y)
accel_z_avg = np.mean(accel_z)

accel_x_std = np.std(accel_x)
accel_y_std = np.std(accel_y)
accel_z_std = np.std(accel_z)

gyro_x = np.array(gyro_x)
gyro_y = np.array(gyro_y)
gyro_z = np.array(gyro_z)

gyro_x_avg = np.mean(gyro_x)
gyro_y_avg = np.mean(gyro_y)
gyro_z_avg = np.mean(gyro_z)

gyro_x_std = np.std(gyro_x)
gyro_y_std = np.std(gyro_y)
gyro_z_std = np.std(gyro_z)

print("Calibrated IMU results:")
print(f"Accel X = {accel_x_avg} +/- {accel_x_std} G's")
print(f"Accel Y = {accel_y_avg} +/- {accel_y_std} G's")
print(f"Accel Z = {accel_z_avg} +/- {accel_z_std} G's")
print(f"Gyro X = {gyro_x_avg} +/- {gyro_x_std} deg/s")
print(f"Gyro Y = {gyro_y_avg} +/- {gyro_y_std} deg/s")
print(f"Gyro Z = {gyro_z_avg} +/- {gyro_z_std} deg/s")
print()

print("Final offsets:")
print(f"Accel X = {accel_x_offset}")
print(f"Accel Y = {accel_y_offset}")
print(f"Accel Z = {accel_z_offset}")
print(f"Gyro X = {gyro_x_offset}")
print(f"Gyro Y = {gyro_y_offset}")
print(f"Gyro Z = {gyro_z_offset}")
print()

y = input("Press any key to start streaming data.")
while True:
	a_d = imu.get_accel_data(keep_int=False)
	g_d = imu.get_gyro_data(keep_int=False)
	a_d = [round(d, 3) for d in a_d]
	g_d = [round(d, 3) for d in g_d]
	print("{:>7}, {:>7}, {:>7}, {:>8}, {:>8}, {:>8}".format(a_d[0], a_d[1], a_d[2], g_d[0], g_d[1], g_d[2]))
	time.sleep(0.005)