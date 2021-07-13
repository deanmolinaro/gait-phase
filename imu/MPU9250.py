import board
import busio

class MPU9250:
	# MPU9250 I2C address
	MPU9250_ADDR = 0x68

	# POWER SETTINGS
	PWR_MGMT_1 = 0x6B
	H_RESET_OR = 0x80 # Resets all IMU registers

	# CONFIGURE ACCELEROMETER
	# Accel sensitivity settings
	ACCEL_CONFIG = 0x1C # Register to change accel scale
	ACCEL_FS_SEL = 0x18 # Bit mask [4:3] for accel scale from ACCEL_CONFIG
	ACCEL_2G_OR = 0x00 # [4:3]->00
	ACCEL_2G_AND = 0xE7 # [4:3]->00
	ACCEL_4G_OR = 0x08 # [4:3]->01
	ACCEL_4G_AND = 0xEF # [4:3]->01
	ACCEL_8G_OR = 0x10 # [4:3]->10
	ACCEL_8G_AND = 0xF7 # [4:3]->10
	ACCEL_16G_OR = 0x18 # [4:3]->11
	ACCEL_16G_AND = 0xFF # [4:3]->11

	# Accel offset settings
	XA_OFFSET_H = 0x77
	XA_OFFSET_L = 0x78
	YA_OFFSET_H = 0x7A
	YA_OFFSET_L = 0x7B
	ZA_OFFSET_H = 0x7D
	ZA_OFFSET_L = 0x7E

	# Accel digital lowpass filter settings
	ACCEL_CONFIG_2 = 0x1D
	ACCEL_FCHOICE_B_MASK = 0x08
	ACCEL_FCHOICE_B_DISABLE = 0x00
	ACCEL_FCHOICE_B_ENABLE = 0x01
	A_DLPFCFG_MASK = 0x07
	A_DLPFCFG_218_1HZ = 0x00
	A_DLPFCFG_99HZ = 0x02
	A_DLPFCFG_44_8HZ = 0x03
	A_DLPFCFG_21_2HZ = 0x04
	A_DLPFCFG_10_2HZ = 0x05
	A_DLPFCFG_5_05HZ = 0x06
	A_DLPFCFG_420HZ = 0x07

	# Accel measurements
	ACCEL_XOUT_H = 0x3B
	ACCEL_XOUT_L = 0x3C
	ACCEL_YOUT_H = 0x3D
	ACCEL_YOUT_L = 0x3E
	ACCEL_ZOUT_H = 0x3F
	ACCEL_ZOUT_L = 0x40

	# CONFIGURE GYROSCOPE
	GYRO_CONFIG = 0x1B

	# Gyro senstivity settings
	GYRO_FS_SEL = 0x18 # Bit mas [4:3] for gyro scale from GYRO_CONFIG
	GYRO_250DPS_OR = 0x00 # [4:3]->00
	GYRO_250DPS_AND = 0xE7 # [4:3]->00
	GYRO_500DPS_OR = 0x08 # [4:3]->01
	GYRO_500DPS_AND = 0xEF # [4:3]->01
	GYRO_1000DPS_OR = 0x10 # [4:3]->10
	GYRO_1000DPS_AND = 0xF7 # [4:3]->10
	GYRO_2000DPS_OR = 0x18 # [4:3]->11
	GYRO_2000DPS_AND = 0xFF # [4:3]->11

	# Gyro offset settings
	XG_OFFSET_H = 0x13
	XG_OFFSET_L = 0x14
	YG_OFFSET_H = 0x15
	YG_OFFSET_L = 0x16
	ZG_OFFSET_H = 0x17
	ZG_OFFSET_L = 0x18

	# Gyro measurements
	GYRO_XOUT_H = 0x43
	GYRO_XOUT_L = 0x44
	GYRO_YOUT_H = 0x45
	GYRO_YOUT_L = 0x46
	GYRO_ZOUT_H = 0x47
	GYRO_ZOUT_L = 0x48

	# General masks
	FIRST_EIGHT = 0xFF00
	LAST_EIGHT = 0xFF
	FIRST_SEVEN = 0xFE00
	LAST_SEVEN = 0x7F
	MAX_15BIT = 0x7FFF
	MAX_16BIT = 0xFFFF

	def __init__(self, scl=board.SCL, sda=board.SDA):
		self.i2c = busio.I2C(scl, sda)
		self.accel_scale = 4/(2**16) # Default accelerometer sensitivity is +/-2G
		self.gyro_scale = 500/(2**16) # Default gyroscope sensitivity is +/-250dps
		self.reset_registers()


	def reset_registers(self):
		reg_data = bytearray(1)
		self.i2c.writeto(self.MPU9250_ADDR, bytes([self.PWR_MGMT_1]), stop=False)
		self.i2c.readfrom_into(self.MPU9250_ADDR, reg_data)
		reg_data = reg_data[0]
		reg_data |= self.H_RESET_OR
		self.i2c.writeto(self.MPU9250_ADDR, bytes([self.PWR_MGMT_1, reg_data]), stop=False)
		return 1

	# Accelerometer Methods
	def get_accel_config(self):
		reg_data = bytearray(1)
		self.i2c.writeto(self.MPU9250_ADDR, bytes([self.ACCEL_CONFIG]), stop=False)
		self.i2c.readfrom_into(self.MPU9250_ADDR, reg_data)
		return reg_data[0]

	def get_accel_data(self, keep_int=False):
		reg_data = bytearray(6)
		self.i2c.writeto(self.MPU9250_ADDR, bytes([self.ACCEL_XOUT_H]), stop=False)
		self.i2c.readfrom_into(self.MPU9250_ADDR, reg_data)
		accel_x = (reg_data[0] << 8 | reg_data[1]) - ((reg_data[0] >> 7)*(2**16))
		accel_y = (reg_data[2] << 8 | reg_data[3]) - ((reg_data[2] >> 7)*(2**16))
		accel_z = (reg_data[4] << 8 | reg_data[5]) - ((reg_data[4] >> 7)*(2**16))

		if not keep_int:
			accel_x *= self.accel_scale
			accel_y *= self.accel_scale
			accel_z *= self.accel_scale

		return accel_x, accel_y, accel_z

	def get_accel_lowpass(self):
		reg_data = bytearray(1)
		self.i2c.writeto(self.MPU9250_ADDR, bytes([self.ACCEL_CONFIG_2]))
		self.i2c.readfrom_into(self.MPU9250_ADDR, reg_data)
		reg_data = reg_data[0]
		accel_fchoice_b = (reg_data & self.ACCEL_FCHOICE_B_MASK) >> 3
		dlpf = reg_data & self.A_DLPFCFG_MASK
		return accel_fchoice_b, dlpf

	def set_accel_lowpass(self, config):
		self.i2c.writeto(self.MPU9250_ADDR, bytes([self.ACCEL_CONFIG_2, config]))
		return

	def set_accel_lowpass_mode(self, mode):
		accel_fchoice_b, _ = get_accel_lowpass()
		if mode==0:
			a_dlpfcfg = self.A_DLPFCFG_218_1HZ
		elif mode==1:
			a_dlpfcfg = self.A_DLPFCFG_218_1HZ
		elif mode==2:
			a_dlpfcfg = self.A_DLPFCFG_99HZ
		elif mode==3:
			a_dlpfcfg = self.A_DLPFCFG_44_8HZ
		elif mode==4:
			a_dlpfcfg = self.A_DLPFCFG_21_2HZ
		elif mode==5:
			a_dlpfcfg = self.A_DLPFCFG_10_2HZ
		elif mode==6:
			a_dlpfcfg = self.A_DLPFCFG_5_05HZ
		elif mode==7:
			a_dlpfcfg = self.A_DLPFCFG_420HZ
		else:
			print('Warning - Unknown mode. DLPF not set.')
			return
		config = (accel_fchoice_b << 3) | a_dlpfcfg
		set_accel_lowpass(config)

	def enable_accel_lowpass(self):
		_, dlpf = get_accel_lowpass()
		config = (self.ACCEL_FCHOICE_B_DISABLE << 3) | dlpf
		set_accel_lowpass(config)
		return

	def disable_accel_lowpass(self):
		_, dlpf = get_accel_lowpass()
		config = (self.ACCEL_FCHOICE_B_ENABLE << 3) | dlpf 
		set_accel_lowpass(config)
		return

	def get_accel_offset(self, reg_address_H, res_bit=False):
		reg_data = bytearray(2)
		self.i2c.writeto(self.MPU9250_ADDR, bytes([reg_address_H]), stop=False)
		self.i2c.readfrom_into(self.MPU9250_ADDR, reg_data)
		if not res_bit:
			return (reg_data[0] << 7) | (reg_data[1] >> 1) - ((reg_data[0] >> 7)*(2**15))
		else:
			return (reg_data[0] << 7) | (reg_data[1] >> 1) - ((reg_data[0] >> 7)*(2**15)), reg_data[1] & 0x01

	def get_accel_offset_x(self, res_bit=False):
		return self.get_accel_offset(self.XA_OFFSET_H, res_bit=res_bit)

	def get_accel_offset_y(self, res_bit=False):
		return self.get_accel_offset(self.YA_OFFSET_H, res_bit=res_bit)

	def get_accel_offset_z(self, res_bit=False):
		return self.get_accel_offset(self.ZA_OFFSET_H, res_bit=res_bit)

	def set_accel_offset(self, offset, reg_address_H):
		if offset < 0:
			offset_bin = abs(offset)
			offset_bin ^= self.MAX_15BIT
			offset_bin += 0x01 
			offset_H = offset_bin >> 7
			offset_L = (offset_bin & self.LAST_SEVEN) << 1
		else:
			offset_H = offset >> 7
			offset_L = (offset & self.LAST_SEVEN) << 1

		self.i2c.writeto(self.MPU9250_ADDR, bytes([reg_address_H, offset_H, offset_L]), stop=False)

		if offset != self.get_accel_offset(reg_address_H):
			return 0
		return 1

	# def set_accel_offset(self, offset, reg_address_H):  # TODO: Handle negative numbers using two's complement (see set_gyro_offset)
	# 	offset_orig, res_bit = self.get_accel_offset(reg_address_H, res_bit=True)
	# 	offset_H = offset >> 7
	# 	offset_L = ((offset & self.LAST_SEVEN) << 1) | res_bit

	# 	self.i2c.writeto(self.MPU9250_ADDR, bytes([reg_address_H, offset_H, offset_L]), stop=False)
	# 	offset_new = self.get_accel_offset(reg_address_H)

	# 	if offset != offset_new:
	# 		return 0
	# 	return 1

	def set_accel_offset_x(self, offset):
		if not self.set_accel_offset(offset, self.XA_OFFSET_H):
			print('Warning - Accel X offset not set!')
			return 0
		else:
			print('Accel X offset set to ' + str(offset))
			return 1

	def set_accel_offset_y(self, offset):
		if not self.set_accel_offset(offset, self.YA_OFFSET_H):
			print('Warning - Accel Y offset not set!')
			return 0
		else:
			print('Accel Y offset set to ' + str(offset))
			return 1

	def set_accel_offset_z(self, offset):
		if not self.set_accel_offset(offset, self.ZA_OFFSET_H):
			print('Warning - Accel Z offset not set!')
			return 0
		else:
			print('Accel Z offset set to ' + str(offset))
			return 1

	def set_accel_scale(self, scale):
		accel_config = self.get_accel_config()

		if scale==2:
			accel_config |= self.ACCEL_2G_OR
			accel_config &= self.ACCEL_2G_AND
		elif scale==4:
			accel_config |= self.ACCEL_4G_OR
			accel_config &= self.ACCEL_4G_AND
		elif scale==8:
			accel_config |= self.ACCEL_8G_OR
			accel_config &= self.ACCEL_8G_AND
		elif scale==16:
			accel_config |= self.ACCEL_16G_OR
			accel_config &= self.ACCEL_16G_AND
		else:
			print('Warning - Requested accel scale does not exist!')
			return 0

		self.i2c.writeto(self.MPU9250_ADDR, bytes([self.ACCEL_CONFIG, accel_config]), stop=False)
		accel_config_new = self.get_accel_config()

		if accel_config != accel_config_new:
			print('Warning - Accel scale not set!')
			return 0
		else:
			self.accel_scale = (2*scale)/(2**16)
			print('Accel scale set to +/-' + str(scale) + 'G')
			return 1


	# Gyroscope Methods
	def get_gyro_config(self):
		reg_data = bytearray(1)
		self.i2c.writeto(self.MPU9250_ADDR, bytes([self.GYRO_CONFIG]), stop=False)
		self.i2c.readfrom_into(self.MPU9250_ADDR, reg_data)
		return reg_data[0]

	def get_gyro_data(self, keep_int=False):
		reg_data = bytearray(6)
		self.i2c.writeto(self.MPU9250_ADDR, bytes([self.GYRO_XOUT_H]), stop=False)
		self.i2c.readfrom_into(self.MPU9250_ADDR, reg_data)
		gyro_x = (reg_data[0] << 8 | reg_data[1]) - ((reg_data[0] >> 7)*(2**16))
		gyro_y = (reg_data[2] << 8 | reg_data[3]) - ((reg_data[2] >> 7)*(2**16))
		gyro_z = (reg_data[4] << 8 | reg_data[5]) - ((reg_data[4] >> 7)*(2**16))

		if not keep_int:
			gyro_x *= self.gyro_scale
			gyro_y *= self.gyro_scale
			gyro_z *= self.gyro_scale

		return gyro_x, gyro_y, gyro_z

	def get_gyro_offset(self, reg_address_H):
		reg_data = bytearray(2)
		self.i2c.writeto(self.MPU9250_ADDR, bytes([reg_address_H]), stop=False)
		self.i2c.readfrom_into(self.MPU9250_ADDR, reg_data)
		return ((reg_data[0] << 8) | reg_data[1]) - ((reg_data[0] >> 7)*(2**16))

	def get_gyro_offset_x(self):
		return self.get_gyro_offset(self.XG_OFFSET_H)

	def get_gyro_offset_y(self):
		return self.get_gyro_offset(self.YG_OFFSET_H)

	def get_gyro_offset_z(self):
		return self.get_gyro_offset(self.ZG_OFFSET_H)

	def set_gyro_offset(self, offset, reg_address_H):
		if offset < 0:
			offset_bin = abs(offset)
			offset_bin ^= self.MAX_16BIT
			offset_bin += 0x01
			offset_H = offset_bin >> 8
			offset_L = offset_bin & self.LAST_EIGHT
		else:
			offset_H = offset >> 8
			offset_L = offset & self.LAST_EIGHT
		self.i2c.writeto(self.MPU9250_ADDR, bytes([reg_address_H, offset_H, offset_L]), stop=False)
		# offset_new = get_gyro_offset(reg_address_H)

		if offset != self.get_gyro_offset(reg_address_H):
			return 0
		return 1

	def set_gyro_offset_x(self, offset):
		if not self.set_gyro_offset(offset, self.XG_OFFSET_H):
			print('Warning - Gyro X offset not set!')
			return 0
		else:
			print('Gyro X offset set to ' + str(offset))
			return 1

	def set_gyro_offset_y(self, offset):
		if not self.set_gyro_offset(offset, self.YG_OFFSET_H):
			print('Warning - Gyro Y offset not set!')
			return 0
		else:
			print('Gyro Y offset set to ' + str(offset))
			return 1

	def set_gyro_offset_z(self, offset):
		if not self.set_gyro_offset(offset, self.ZG_OFFSET_H):
			print('Warning - Gyro Z offset not set!')
			return 0
		else:
			print('Gyro Z offset set to ' + str(offset))
			return 1

	def set_gyro_scale(self, scale):
		gyro_config = self.get_gyro_config()

		if scale==250:
			gyro_config |= self.GYRO_250DPS_OR
			gyro_config &= self.GYRO_250DPS_AND
		elif scale==500:
			gyro_config |= self.GYRO_500DPS_OR
			gyro_config &= self.GYRO_500DPS_AND
		elif scale==1000:
			gyro_config |= self.GYRO_1000DPS_OR
			gyro_config &= self.GYRO_1000DPS_AND
		elif scale==2000:
			gyro_config |= self.GYRO_2000DPS_OR
			gyro_config &= self.GYRO_2000DPS_AND
		else:
			print('Warning - Requested gyro scale does not exist!')
			return 0

		self.i2c.writeto(self.MPU9250_ADDR, bytes([self.GYRO_CONFIG, gyro_config]), stop=False)
		gyro_config_new = self.get_gyro_config()

		if gyro_config_new != gyro_config:
			print('Warning - Gyro scale was not set!')
			return 0
		else:
			self.gyro_scale = (2*scale)/(2**16)
			print('Gyro scale set to +/-' + str(scale) + 'DPS')
			return 1

	def get_imu_data(self, keep_int=False):
		reg_data = bytearray(14)
		self.i2c.writeto(self.MPU9250_ADDR, bytes([self.ACCEL_XOUT_H]), stop=False)
		self.i2c.readfrom_into(self.MPU9250_ADDR, reg_data)
		accel_x = (reg_data[0] << 8 | reg_data[1]) - ((reg_data[0] >> 7)*(2**16))
		accel_y = (reg_data[2] << 8 | reg_data[3]) - ((reg_data[2] >> 7)*(2**16))
		accel_z = (reg_data[4] << 8 | reg_data[5]) - ((reg_data[4] >> 7)*(2**16))

		gyro_x = (reg_data[8] << 8 | reg_data[9]) - ((reg_data[8] >> 7)*(2**16))
		gyro_y = (reg_data[10] << 8 | reg_data[11]) - ((reg_data[10] >> 7)*(2**16))
		gyro_z = (reg_data[12] << 8 | reg_data[13]) - ((reg_data[12] >> 7)*(2**16))

		if not keep_int:
			accel_x *= self.accel_scale
			accel_y *= self.accel_scale
			accel_z *= self.accel_scale

			gyro_x *= self.gyro_scale
			gyro_y *= self.gyro_scale
			gyro_z *= self.gyro_scale

		return accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z
