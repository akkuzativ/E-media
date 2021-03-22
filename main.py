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

# arr = []
data = []
f = open(file="sine440.wav", mode="rb")

while 1:
    id = bytes.decode(f.read(4))
    if len(id):
        size = int.from_bytes(f.read(4), byteorder="little")
        # print(size)
    else:
        break
    if id == "RIFF":
        data = [bytes.decode(f.read(4))]
        riffChunk = Chunk(id, size, data)
        riffChunk.print()
    elif id == "fmt ":
        data = [int.from_bytes(f.read(2), byteorder="little"), int.from_bytes(f.read(2), byteorder="little"), int.from_bytes(f.read(4), byteorder="little"), int.from_bytes(f.read(4), byteorder="little"), int.from_bytes(f.read(2), byteorder="little"), int.from_bytes(f.read(2), byteorder="little")]
        fmtChunk = Chunk(id, size, data)
        fmtChunk.print()
    elif id == "INFO":
        for i in range(int((size-8)/2)):
            data.append(int.from_bytes(f.read(4), byteorder="little"))
        infoChunk = Chunk(id, size, data)
        infoChunk.print()
    elif id == "data":
        # for i in range(int(size / 2)):
        #     data.append(int.from_bytes(f.read(4), byteorder="little"))
        dataChunk = Chunk(id, size, 0) # TODO: wczytać dane do dataChunk.data
        dataChunk.print()
        break # tymczasowe wyjście z pętli, bo UnicodeDecodeError

sample_len = int(fmtChunk.size / 8)
samples = []
data = f.read(-1)

pos = 0
while pos < len(data):
    samples.append(int.from_bytes(data[pos:pos+sample_len], byteorder="little", signed=True))
    pos = pos + sample_len

plt.plot(list(range(0, len(samples), 1))[0:110], samples[0:110])
plt.show()

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