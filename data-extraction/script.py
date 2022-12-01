import codecs
import struct
import csv

class Epoch:
	def __init__(self, bytes):
		state = chr(bytes[0]
		EEGv = struct.unpack('f', bytes[-12:-8])[0]
		EMGv = struct.unpack('f', bytes[-8:-4])[0]
		temp = struct.unpack('f', chunk[-4:])[0]

	def write_row(thewriter):
		thewriter.writerow([chr(chunk[0]), struct.unpack('f', chunk[-12:-8])[0], struct.unpack('f', chunk[-8:-4])[0], struct.unpack('f', chunk[-4:])[0]])


chunk_size = 1617 # 1 state byte + 401 DC-component of the EEG + EEG variability, the EMG variability and the temperature of the mouse

output_file = '../data/10101.csv'
input_file = '../data/10101.smo'

with open(output_file, 'w', newline='') as csvfile:
	thewriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
	thewriter.writerow(["state", "EEGv", "EMGv", "temp"])
	with open(input_file, "rb") as in_file:
		while True:
			chunk = in_file.read(chunk_size)
			if len(chunk) < 1617:
				break
			epoch = Epoch(chunk)
			epoch.write_row(thewriter)

