from matplotlib import pyplot as plt


class Chunk:
    def __init__(self, id, size, data):
        self.id = id
        self.size = size
        self.data = data

    def print(self):
        print(self.id)
        print(self.size)
        print(self.data)
    pass


class WavFile:
    def __init__(self, a, b, c):
        self.riff_descriptor = a
        self.fmt = b
        self.data = c
    pass

sample_len = 0
f = open(file="sine440.wav", mode="rb")

while 1:
    id = bytes.decode(f.read(4))
    if len(id):
        size = int.from_bytes(f.read(4), byteorder="little")
        # print(size)
    else:
        break
    data = []
    if id == "RIFF":
        data = [bytes.decode(f.read(4))]
        riffChunk = Chunk(id, size, data)
        riffChunk.print()
    elif id == "fmt ":
        sample_len = int(size / 8)
        data = [int.from_bytes(f.read(2), byteorder="little"), int.from_bytes(f.read(2), byteorder="little"), int.from_bytes(f.read(4), byteorder="little"), int.from_bytes(f.read(4), byteorder="little"), int.from_bytes(f.read(2), byteorder="little"), int.from_bytes(f.read(2), byteorder="little")]
        fmtChunk = Chunk(id, size, data)
        fmtChunk.print()
    elif id == "INFO":
        for i in range(int((size-8)/2)):
            data.append(int.from_bytes(f.read(4), byteorder="little"))
        infoChunk = Chunk(id, size, data)
        infoChunk.print()
    elif id == "data":
        for i in range(int(size / sample_len)):
            data.append(int.from_bytes(f.read(sample_len), byteorder="little", signed=True))
        dataChunk = Chunk(id, size, data)
        # dataChunk.print()

samples = []

pos = 0
while pos < len(dataChunk.data):
    samples.append(dataChunk.data[pos])
    pos += 1

plt.plot(list(range(0, len(samples), 1))[0:110], samples[0:110])
plt.show()

# arr = []
# for i in range(5):
#     arr.append(f.read(4))
# for i in range(2):
#     arr.append(f.read(2))
# for i in range(2):
#     arr.append(f.read(4))
# for i in range(2):
#     arr.append(f.read(2))
# for i in range(2):
#     arr.append(f.read(4))
#
# for i in [0, 2, 3, 11]:
#     arr[i] = bytes.decode(arr[i])
# for i in [1, 4, 5, 6, 7, 8, 9, 10, 12]:
#     arr[i] = int.from_bytes(arr[i], byteorder="little")
# print(arr)
