from matplotlib import pyplot as plt
import struct
import audioop
import numpy as np


class Chunk:
    def __init__(self, id: str, size: int, data=None):
        self.id = id
        self.size = size
        self.data = data

    def __repr__(self):
        return str(self.id) + " - " + str(self.size)# + "\n" + str(self.data)
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
                return str(self.audio_format) + " - " + str(self.num_channels) + " - " + str(self.sample_rate) +\
                       " - " + str(self.byte_rate) + " - " + str(self.block_align) + " - " + str(self.bits_per_sample) +\
                       " - " + str(self.num_extra_format_bytes) + " - " + str(self.extra_format_bytes)
            else:
                return str(self.audio_format) + " - " + str(self.num_channels) + " - " + str(self.sample_rate) +\
                       " - " + str(self.byte_rate) + " - " + str(self.block_align) + " - " + str(self.bits_per_sample)
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
        return Chunk.__repr__(self) + " - " + str(self.data)
    pass


class INFOChunk(Chunk):
    class Contents: #TODO do rozszerzenia o pozostałe
        IART: INFOsubChunk #wykonawca
        INAM: INFOsubChunk #tytuł utworu
        IPRD: INFOsubChunk #tytuł albumu
        ICRD: INFOsubChunk #data wydania
        IGNR: INFOsubChunk # gatunek
        ICMT: INFOsubChunk #komentarze
        ITRK: INFOsubChunk #komentarze
        ISFT: INFOsubChunk #oprogramowanie

        def __repr__(self):
            return str(self.IART) + " - " + str(self.INAM) + " - " + str(self.IPRD) + \
                   " - " + str(self.ICRD) + " - " + str(self.IGNR) + " - " + str(self.ICMT) + " - " + str(self.ITRK)

            # output=None
            # var = ""
            # for field in range(7):
            #     if dir(self)[field] is not None:
            #         var = "str(self."+ dir(self)[field]+")"
            #         print(var)
            #         # print(output)
            #         output += str(globals()[var])
            #     if field <6:
            #         var+=" + \" - \" +"
            # exec(var)
            # print(output)
            # return # + " - " + str(self.ICMT)

    data: Contents

    def __init__(self, id: str, size: int, data: list):
        Chunk.__init__(self=self, id=id, size=size, data=None)
        self.data = INFOChunk.Contents()
        start = 0
        while start < len(data)-1:
            subid = bytes.decode(data[start:start+4])
            if len(subid):
                sizesubid = int.from_bytes(data[start+4:start+8], byteorder="little")
            if subid in self.Contents.__annotations__:
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
        labl: None

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
        return Chunk.__repr__(self) + "\n" + str(self.data)
    pass

class ID3Chunk(Chunk):
    class Contents: #TODO do rozszerzenia o pozostałe
        TPE1: INFOsubChunk #wykonawca
        COMM: INFOsubChunk #tytuł utworu
        TIT2: INFOsubChunk #tytuł albumu
        TDRC: INFOsubChunk # gatunek
        TALB: INFOsubChunk #komentarze
        TRCK: INFOsubChunk #oprogramowanie
        TCON: INFOsubChunk #oprogramowanie


        def __repr__(self):
            return str(self.TPE1) + "\n" + str(self.TIT2) + "\n" + str(self.COMM) + \
                   "\n" + str(self.TALB) #+ "\n" + str(self.TDRC) + "\n" + str(self.TRCK) \
                   # + "\n" + str(self.TCON)

    data: Contents

    def __init__(self, id: str, size: int, data: list):
        Chunk.__init__(self=self, id=id, size=size, data=None)
        self.data = ID3Chunk.Contents()
        start = 0
        while len(data) > 45:
            subid = bytes.decode(data[start:start+4])
            # if len(subid):
                # sizesubid = int.from_bytes(data[start+4:start+8], byteorder="little")
            if subid in self.Contents.__annotations__:
                for byte in data[5:-3]:
                    # print(bytes.decode(data[data.index(byte)+5:data.index(byte)+9]))
                    if bytes.decode(data[data.index(byte)+5:data.index(byte)+9]) in self.Contents.__annotations__:
                        sizesubid=data.index(byte)+5-4 # TODO nie wszystkie chunki są rozpoznawane
                        # print(bytes.decode(data[data.index(byte)+5:data.index(byte)+9]))
                        # print(data.index(byte)+5)
                        exec("%s=%s" % ('self.data.' + subid, "INFOsubChunk(subid,data.index(byte)+5-4, data[start+4:start+data.index(byte)+5])"))
                        data = data[data.index(byte) + 5:]
                        break
                # print(data)
            else:
                unrecognizedChunk = Chunk(subid, size-4, data[start+4:start+size])  # TODO niezapisywany
                print(unrecognizedChunk)

    def __repr__(self):
        return Chunk.__repr__(self) + "\n" + str(self.data)
    pass


class id3Chunk(Chunk):
    class Contents:
        ID3: ID3Chunk
        def __repr__(self):
            return str(self.ID3)
        pass

    data: Contents

    def __init__(self, id: str, size: int, data: [bytes, int]):
        Chunk.__init__(self=self, id=id, size=size, data=None)
        self.data = id3Chunk.Contents()
        start = 0
        while start < data.index(data[-1]):
            subid = bytes.decode(data[start:start+3])
            if len(subid):
                subsize = int.from_bytes(data[start+3:start+10], byteorder="little")
                # print(subsize)
            if subid == "ID3":
                # print(data)
                self.data.ID3=ID3Chunk(subid, size - 10, data[start+10:start+size])
            else:
                unrecognizedChunk = Chunk(subid, size-10, data[start+10:start+size])  # TODO niezapisywany
                print(unrecognizedChunk)
            start = start + size

    def __repr__(self):
        return Chunk.__repr__(self) + "\n" + str(self.data)
    pass

class WavFile:
    def __init__(self, a: RIFFHeader = None, b: FmtChunk = None, c: DataChunk = None, d: LISTChunk = None, e: list = None):
        self.riff_descriptor = a
        self.fmt = b
        self.data = c
        self.list = d
        self.unrecognizedChunks = e

    pass

size = 0
sample_len = 0
f = open(file="data/sine440-list.wav", mode="rb")
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
    elif id == "id3 ":
        data = [f.read(size)]
        id3Chunk = id3Chunk(id, size, data[0])
        print(id3Chunk)
    elif id == "data":
        sample_len = int(fmtChunk.data.bits_per_sample / 8)

        # wersja klasyczna
        def sample_conversion_native(sample):
            if fmtChunk.data.audio_format == 1:
                if sample_len == 1:
                    return int.from_bytes(sample, byteorder="little", signed=False)
                else:
                    return int.from_bytes(sample, byteorder="little", signed=True)
            elif fmtChunk.data.audio_format == 3:
                if sample_len == 4:
                    return struct.unpack("f", sample)[0]
                else:
                    return struct.unpack("d", sample)[0]
            elif fmtChunk.data.audio_format == 6:
                return int.from_bytes(audioop.alaw2lin(sample, sample_len), byteorder="little", signed=True)
            elif fmtChunk.data.audio_format == 7:
                return int.from_bytes(audioop.ulaw2lin(sample, sample_len), byteorder="little", signed=True)
            elif fmtChunk.data.audio_format == 65534:
                pass


        # Wersja przyspieszona wykorzystująca numpy
        def sample_conversion_fast(samples: bytes):
            if fmtChunk.data.audio_format == 1:
                if sample_len == 1:
                    return np.frombuffer(samples, "uint8")
                else:
                    return np.frombuffer(samples, "int"+str(sample_len*8))
            elif fmtChunk.data.audio_format == 3:
                if sample_len == 4:
                    return np.frombuffer(samples, "float32")
                else:
                    return np.frombuffer(samples, "float64")
            elif fmtChunk.data.audio_format == 6:
                ret = []
                for i in range(int(size / sample_len)):
                    ret.append(int.from_bytes(audioop.alaw2lin(
                        samples[i * sample_len:i * sample_len + sample_len], sample_len),
                        byteorder="little", signed=True))
                return np.array(ret)
            elif fmtChunk.data.audio_format == 7:
                ret = []
                for i in range(int(size / sample_len)):
                    ret.append(int.from_bytes(audioop.ulaw2lin(
                        samples[i * sample_len:i * sample_len + sample_len], sample_len),
                        byteorder="little", signed=True))
                return np.array(ret)
            elif fmtChunk.data.audio_format == 65534:
                pass

        raw_samples = f.read(size)
        samples = []

        # wykorzystanie klasycznego sposobu wczytywania - iteracje w pętli
        for i in range(int(size/sample_len)):
            samples.append(sample_conversion_native(raw_samples[i*sample_len:i*sample_len+sample_len]))

        # wykorzystanie przyspieszonego wczytywania - zwraca od razu array sampli
        # data = sample_conversion_fast(raw_samples)

        #TODO zdecydować się na sposób wczytywania
        channels = []
        if fmtChunk.data.num_channels > 1:
            for c in range(fmtChunk.data.num_channels):
                channels.append(samples[c::fmtChunk.data.num_channels])
            data = channels
        else:
            data = channels.append(samples)
        dataChunk = DataChunk(id, size, data)
        print(dataChunk)
    else: # TODO factchunk do przerobienia
        data = [f.read(size)]
        unrecognizedChunk = Chunk(id, size, data)
        print(unrecognizedChunk)
        print(unrecognizedChunk.data)

# File = WavFile(riffChunk, fmtChunk, dataChunk, listChunk)

samples = dataChunk.data.samples
#TODO downsampling danych przy rysowaniu wykresów dla usprawnienia obliczenia i czasu rysowania
#plt.plot(range(0, 100, 1), samples[0:100])
plt.plot(range(0, len(samples[0])), samples[0])
plt.show(block=True)
#np.seterr(divide='ignore')  #TODO przy logarytmie wewnatrz tworzenia spektrogramu pojawia sie warning, tymczasowe
#plt.specgram(x=samples, Fs=fmtChunk.data.sample_rate, scale="dB")
#plt.yscale("symlog")
#plt.ylabel("Frequency [Hz]")
#plt.xlabel("Time [s]")
#plt.show(block=True)
