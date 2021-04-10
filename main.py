from matplotlib import pyplot as plt


class Chunk:
    def __init__(self, id="", size=0, data=None):
        self.id = id
        self.size = size
        # self.data = data

    def __repr__(self):
        return str(self.id) + "\n" + str(self.size)# + "\n" + str(self.data)
    pass


class RIFFHeader(Chunk):
    class Contents:
        format: str

        def __init__(self, data: list):
            self.format = data[0]

        def __repr__(self):
            return str(self.format)
        pass
    data: Contents

    def __init__(self, id: str, size: int, data: list):
        Chunk.__init__(self=self, id=id, size=size, data=None)
        self.data = RIFFHeader.Contents(data)

    def __repr__(self):
        return Chunk.__repr__(self) + "\n" + str(self.data)
    pass


class FmtChunk(Chunk):
    class Contents:
        audio_format: int
        num_channels: int
        sample_rate: int
        byte_rate: int
        block_align: int
        bits_per_sample: int

        def __init__(self, data: list):
            self.audio_format = data[0]
            self.num_channels = data[1]
            self.sample_rate = data[2]
            self.byte_rate = data[3]
            self.block_align = data[4]
            self.bits_per_sample = data[5]

        def __repr__(self):
            return str(self.audio_format) + "\n" + str(self.num_channels) + "\n" + str(self.sample_rate) +\
                   "\n" + str(self.byte_rate) + "\n" + str(self.block_align) + "\n" + str(self.bits_per_sample)
        pass

    data: Contents

    def __init__(self, id: str, size: int, data: list):
        Chunk.__init__(self=self, id=id, size=size, data=None)
        self.data = FmtChunk.Contents(data)

    def __repr__(self):
        return Chunk.__repr__(self) + "\n" + str(self.data)
    pass


class ISFTChunk(Chunk):
    class Contents:
        Lavf: str
        number: str

        def __init__(self, data: list):
            self.Lavf = data[0]
            self.number = data[1]

        def __repr__(self):
            return str(self.Lavf) + "\n" + str(self.number)
        pass

    data: Contents

    def __init__(self, id: str, size: int, data: list):
        Chunk.__init__(self=self, id=id, size=size, data=None)
        id2 = bytes.decode(data[0:4])
        id3 = bytes.decode(data[4:])
        self.data = ISFTChunk.Contents([id2, id3])

    def __repr__(self):
        return Chunk.__repr__(self) + "\n" + str(self.data)
    pass


class INFOChunk(Chunk):
    class Contents:
        ISFT: ISFTChunk

        def __init__(self, data: ISFTChunk):
            self.ISFT = data

        def __repr__(self):
            return str(self.ISFT)

    data: Contents

    def __init__(self, id: str, size: int, data: list):
        Chunk.__init__(self=self, id=id, size=size, data=None)
        subid = bytes.decode(data[0:4])
        if len(subid):
            sizesubid = int.from_bytes(data[4:8], byteorder="little")
        if subid == "ISFT":
            data = data[8:]
        isftChunk = ISFTChunk(subid, sizesubid, data)
        self.data = INFOChunk.Contents(isftChunk)

    def __repr__(self):
        return Chunk.__repr__(self) + "\n" + str(self.data)
    pass


class LISTChunk(Chunk):
    class Contents:
        INFO: INFOChunk

        def __init__(self, data: INFOChunk):
            self.INFO = data

        def __repr__(self):
            return str(self.INFO)
        pass

    data: Contents

    def __init__(self, id: str, size: int, data: list):
        Chunk.__init__(self=self, id=id, size=size, data=None)
        subid = bytes.decode(data[0][0:4])
        if subid == "INFO":
            data = data[0][4:]
            infoChunk = INFOChunk(subid, size-4, data)
        self.data = LISTChunk.Contents(infoChunk)

    def __repr__(self):
        return Chunk.__repr__(self) + "\n" + str(self.data)
    pass


class DataChunk(Chunk):
    class Contents:
        samples: list

        def __init__(self, data: list):
            self.samples = data

        def __repr__(self):
            return str(self.samples)
        pass
    data: Contents

    def __init__(self, id: str, size: int, data: list):
        Chunk.__init__(self=self, id=id, size=size, data=None)
        self.data = DataChunk.Contents(data)

    def __repr__(self):
        return Chunk.__repr__(self) + "\n"# + str(self.data)
    pass


class WavFile:
    def __init__(self, a: RIFFHeader = None, b: FmtChunk = None, c: DataChunk = None, d: LISTChunk = None, e: list = None):
        self.riff_descriptor = a
        self.fmt = b
        self.data = c
        self.list = d
        self.unrecognizedChunks = e

    pass


sample_len = 0
f = open(file="Gate of Steiner.wav", mode="rb")
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
        riffChunk = RIFFHeader(id, size, data)
        print(riffChunk)
    elif id == "fmt ":
        sample_len = int(size / 8)
        data = [int.from_bytes(f.read(2), byteorder="little"), int.from_bytes(f.read(2), byteorder="little"),
                int.from_bytes(f.read(4), byteorder="little"), int.from_bytes(f.read(4), byteorder="little"),
                int.from_bytes(f.read(2), byteorder="little"), int.from_bytes(f.read(2), byteorder="little")]
        fmtChunk = FmtChunk(id, size, data)
        print(fmtChunk)
    elif id == "LIST":
        data = [f.read(size)]
        listChunk = LISTChunk(id, size, data)
        print(listChunk)
    elif id == "data":
        # if fmtChunk is None:
        # domyślnie unrecognized lub próba konwersji do 2-bajtowego inta
        # else:
        # obsługa w zależności od fmtChunk.data.audio_format; 1 to 2-bajtowy int
        for i in range(int(size / sample_len)):
            data.append(int.from_bytes(f.read(sample_len), byteorder="little", signed=True))
        dataChunk = DataChunk(id, size, data)
        print(dataChunk)
    else:
        data = [f.read(size)]
        unrecognizedChunk = Chunk(id, size, data)
        print(unrecognizedChunk)

# File = WavFile(riffChunk, fmtChunk, dataChunk, listChunk)

# samples = []
# pos = 0
# while pos < len(dataChunk.data):
#     samples.append(dataChunk.data[pos])
#     pos += 1
# plt.plot(list(range(0, len(samples), 1))[0:110], samples[0:110])
# plt.show()

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
