from matplotlib import pyplot as plt


class Chunk:
    def __init__(self, id: str, size: int, data=None):
        self.id = id
        self.size = size
        self.data = data

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
        num_extra_format_bytes: int
        extra_format_bytes: int

        def __init__(self, data: list):
            self.audio_format = data[0]
            self.num_channels = data[1]
            self.sample_rate = data[2]
            self.byte_rate = data[3]
            self.block_align = data[4]
            self.bits_per_sample = data[5]
            if data.index(data[-1]) > 5:
                self.num_extra_format_bytes = data[6]
                self.extra_format_bytes = data[7]

        def __repr__(self):
            if data.index(data[-1]) > 5:
                return str(self.audio_format) + "\n" + str(self.num_channels) + "\n" + str(self.sample_rate) +\
                       "\n" + str(self.byte_rate) + "\n" + str(self.block_align) + "\n" + str(self.bits_per_sample) +\
                       "\n" + str(self.num_extra_format_bytes) + "\n" + str(self.extra_format_bytes)
            else:
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


class INFOsubChunk(Chunk):
    class Contents:
        data: str

        def __init__(self, data: str):
            self.data = data

        def __repr__(self):
            return str(self.data)
        pass

    data: Contents

    def __init__(self, id: str, size: int, data: list):
        Chunk.__init__(self=self, id=id, size=size, data=None)
        self.data = INFOsubChunk.Contents(bytes.decode(data[:]))

    def __repr__(self):
        return Chunk.__repr__(self) + "\n" + str(self.data)
    pass


class INFOChunk(Chunk):
    class Contents: #TODO do rozszerzenia o pozostałe
        ISFT: INFOsubChunk
        INAM: INFOsubChunk

        # def __init__(self, data: list):  # chyba bezużyteczne
        #     self.ISFT = data[0]
        #     self.INAM = data[1]

        def __repr__(self):
            return str(self.ISFT)# + "\n" + str(self.INAM)

    data: Contents

    def __init__(self, id: str, size: int, data: list):
        Chunk.__init__(self=self, id=id, size=size, data=None)
        self.data = INFOChunk.Contents()
        start = 0
        while  start < data.index(data[-1]):
            subid = bytes.decode(data[start:start+4])
            if len(subid):
                sizesubid = int.from_bytes(data[start+4:start+8], byteorder="little")
            if subid == "ISFT" or subid == "INAM": #TODO do rozszerzenia o pozostałe
                # self.data.ISFT=INFOsubChunk(subid, sizesubid, data[8:sizesubid+8])
                # locals()['self.data.'+subid] = INFOsubChunk(subid, sizesubid, data[8:sizesubid+8])
                exec("%s=%s" % ('self.data.'+subid, "INFOsubChunk(subid, sizesubid, data[start+8:start+sizesubid+8])"))
            else:
                unrecognizedChunk = Chunk(subid, sizesubid, data[start + 8:start + sizesubid + 8])  # TODO niezapisywany
                print(unrecognizedChunk)
            start = start+sizesubid+8

    def __repr__(self):
        return Chunk.__repr__(self) + "\n" + str(self.data)
    pass


class LISTChunk(Chunk):
    class Contents:
        INFO: INFOChunk
        labl: None #TODO jak wyżej

        # def __init__(self, data: INFOChunk):
        #     self.INFO = data

        def __repr__(self):
            return str(self.INFO)
        pass

    data: Contents

    def __init__(self, id: str, size: int, data: [bytes, int]):
        Chunk.__init__(self=self, id=id, size=size, data=None)
        self.data = LISTChunk.Contents()
        start = 0
        while  start < data.index(data[-1]):
            subid = bytes.decode(data[start:start+4])
            if subid == "INFO":
                self.data.INFO=INFOChunk(subid, size-4, data[start+4:start+size])
            else:
                unrecognizedChunk = Chunk(subid, size-4, data[start+4:start+size])  # TODO niezapisywany
                print(unrecognizedChunk)
            start = start + size

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
        riffChunk = RIFFHeader(id, size, data)
        print(riffChunk)
    elif id == "fmt ":
        sample_len = int(size / 8)
        data = [int.from_bytes(f.read(2), byteorder="little"), int.from_bytes(f.read(2), byteorder="little"),
                int.from_bytes(f.read(4), byteorder="little"), int.from_bytes(f.read(4), byteorder="little"),
                int.from_bytes(f.read(2), byteorder="little"), int.from_bytes(f.read(2), byteorder="little")]
        if size > 16:
            data.append(int.from_bytes(f.read(2), byteorder="little"))
            data.append(int.from_bytes(f.read(data[6]), byteorder="little"))
        fmtChunk = FmtChunk(id, size, data)
        print(fmtChunk)
    elif id == "LIST":
        data = [f.read(size)]
        listChunk = LISTChunk(id, size, data[0])
        print(listChunk)
    elif id == "data":
        # if fmtChunk is None:
        # domyślnie unrecognized lub próba konwersji do 2-bajtowego inta
        # else:
        # obsługa w zależności od fmtChunk.data.audio_format; 1 to 2-bajtowy int
        for i in range(int(size / sample_len)):
            data.append(int.from_bytes(f.read(sample_len), byteorder="little", signed=True))
        data.append(int.from_bytes(f.read(2), byteorder="little", signed=True))  # ważne jeżeli data nie jest na końcu
        dataChunk = DataChunk(id, size, data)
        print(dataChunk)
    else: # TODO factchunk do przerobienia
        data = [f.read(size)]
        unrecognizedChunk = Chunk(id, size, data)
        print(unrecognizedChunk)
        print(unrecognizedChunk.data)

# File = WavFile(riffChunk, fmtChunk, dataChunk, listChunk)

samples = []
pos = 0
while pos < len(dataChunk.data.samples):
    samples.append(dataChunk.data.samples[pos])
    pos += 1
plt.plot(list(range(0, len(samples), 1))[0:110], samples[0:110])
plt.show()
