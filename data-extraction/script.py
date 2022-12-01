import codecs
import struct
import csv
import os

class Epoch:
	def __init__(self, bytes):
		self.state = self.state_maping(chr(bytes[0]))
		self.EEGv = struct.unpack('f', bytes[-12:-8])[0]
		self.EMGv = struct.unpack('f', bytes[-8:-4])[0]
		self.temp = struct.unpack('f', chunk[-4:])[0]

	def write_row(self, thewriter):
		thewriter.writerow([self.state, self.EEGv, self.EMGv, self.temp])

	def state_maping(self, state):
		if state == 'w' or '1' or '4':
			return 3
		if state == 'n' or '2' or '5':
			return 2
		if state == 'r' or '3' or '6':
			return 1
		else: 
			return 0
		
chunk_size = 1617 # 1 state byte + 401 DC-component of the EEG + EEG variability, the EMG variability and the temperature of the mouse

dirname = os.path.dirname(__file__)

output_file = '../data/10101.csv'
input_file = '../data/10101.smo'

with open(os.path.join(dirname, output_file), 'w', newline='') as csvfile:
	thewriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
	thewriter.writerow(["state", "EEGv", "EMGv", "temp"])
	with open(input_file, "rb") as in_file:
		while True:
			chunk = in_file.read(chunk_size)
			if len(chunk) < 1617:
				break
			epoch = Epoch(chunk)
			epoch.write_row(thewriter)

