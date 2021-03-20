from matplotlib import pyplot as plt
import numpy as np


f = open(file="sine440.wav", mode="rb")
f.seek(16)
sample_len = int(int.from_bytes(f.read(4), byteorder="little", signed=False)/8)
f.seek(44)
samples = []
data = f.read(-1)

pos = 0
while pos < len(data):
    samples.append(int.from_bytes(data[pos:pos+sample_len], byteorder="little", signed=True))
    pos = pos + sample_len

plt.plot(list(range(0, len(samples), 1))[0:110], samples[0:110])
plt.show()
