from tcpip import ClientTCP
# import numpy as np
import time


client = ClientTCP('192.168.1.2', 8080)

start_time = time.perf_counter()
for i in range(2000):
    for side_str in ['0', '1']:
        accel_x = '%.2f' % int(side_str)
        accel_y = '%.2f' % int(side_str)
        accel_z = '%.2f' % int(side_str)
        gyro_x = '%.2f' % int(side_str)
        gyro_y = '%.2f' % int(side_str)
        gyro_z = '%.2f' % int(side_str)
        ankle_angle = '%.2f' % int(side_str)
        ankle_velocity = '%.2f' % int(side_str)
        message = '!'+side_str + ',' + accel_x+','+accel_y+','+accel_z+','+gyro_x + \
            ','+gyro_y+','+gyro_z+','+ankle_angle+','+ankle_velocity
        # print(message)
        client.to_server(message)

        msg = client.from_server()
        # print(msg)
    time.sleep(0.005)
print(time.perf_counter()-start_time)