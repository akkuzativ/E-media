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
            list = str(self.audio_format) + " - " + str(self.num_channels) + " - " + str(self.sample_rate) +\
                       " - " + str(self.byte_rate) + " - " + str(self.block_align) + " - " + str(self.bits_per_sample)
            if data.index(data[-1]) > 5:
                list += " - " + str(self.num_extra_format_bytes) + " - " + str(self.extra_format_bytes)
            return list
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
        return Chunk.__repr__(self) + " - " + str(self.data) + " - "
    pass


class INFOChunk(Chunk):
    class Contents:
        IART: INFOsubChunk #wykonawca
        INAM: INFOsubChunk #tytuł utworu
        IPRD: INFOsubChunk #tytuł albumu
        ICRD: INFOsubChunk #data wydania
        IGNR: INFOsubChunk # gatunek
        ICMT: INFOsubChunk #komentarze
        ITRK: INFOsubChunk #komentarze
        ISFT: INFOsubChunk #oprogramowanie
        unrecognized: INFOsubChunk

        def __repr__(self):
            list = ""
            try:
                list += str(self.IART)
            except AttributeError:
                pass
            try:
                list += str(self.INAM)
            except AttributeError:
                pass
            try:
                list += str(self.IPRD)
            except AttributeError:
                pass
            try:
                list += str(self.ICRD)
            except AttributeError:
                pass
            try:
                list += str(self.IGNR)
            except AttributeError:
                pass
            try:
                list += str(self.ICMT)
            except AttributeError:
                pass
            try:
                list += str(self.ITRK)
            except AttributeError:
                pass
            try:
                list += str(self.ISFT)
            except AttributeError:
                pass
            try:
                list += str(self.unrecognized)
            finally:
                return list

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
                # unrecognizedChunk = Chunk(subid, sizesubid, data[start + 8:start + sizesubid + 8])
                self.data.unrecognized = INFOsubChunk(subid, sizesubid, data[start + 8:start + sizesubid + 8])
                # print(unrecognizedChunk)
            start = start + sizesubid + 8

        if len(data) > 4 + start:
            self.data.unrecognized = INFOsubChunk(bytes.decode(data[start:start + 4]), len(data)-8, data[start + 4:])

    def __repr__(self):
        return Chunk.__repr__(self) + "\n" + str(self.data)
    pass

class ADTLsubChunk1(Chunk):
    class Contents:
        cueID: str
        sample: str
        purpouse: str
        country: str
        lang: str
        dial: str
        code: str
        data: str

        def __init__(self, cueID: str, data: str, sample: str = None, purpouse: str = None, country: str = None, lang: str = None, dial: str = None, code: str = None):
            self.cueID = cueID
            self.sample = sample
            self.purpouse = purpouse
            self.country = country
            self.lang = lang
            self.dial = dial
            self.code = code
            self.data = data

        def __repr__(self):
            return str(self.cueID) + " - " + str(self.data)
        pass

    data: Contents

    def __init__(self, id: str, size: int, cueID: str, data: list):
        Chunk.__init__(self=self, id=id, size=size, data=None)
        if id != "ltxt":
            self.data = ADTLsubChunk1.Contents(cueID, bytes.decode(data[:]))
        else:
            self.data = ADTLsubChunk1.Contents(cueID, bytes.decode(data[16:]), bytes.decode(data[:4]),
                                               bytes.decode(data[4:8]), bytes.decode(data[8:10]),
                                               bytes.decode(data[10:12]), bytes.decode(data[12:14]),
                                               bytes.decode(data[14:16]))

    def __repr__(self):
        return Chunk.__repr__(self) + " - " + str(self.data)
    pass

class ADTLChunk(Chunk):
    class Contents:
        labl: ADTLsubChunk1
        note: ADTLsubChunk1
        ltxt: ADTLsubChunk1
        unrecognized: ADTLsubChunk1

        def __repr__(self):
            list = ""
            try:
                list += str(self.labl)
            except AttributeError:
                pass
            try:
                list += str(self.note)
            except AttributeError:
                pass
            try:
                list += str(self.ltxt)
            except AttributeError:
                pass
            try:
                list += str(self.unrecognized)
            finally:
                return list

    data: Contents

    def __init__(self, id: str, size: int, data: list):
        Chunk.__init__(self=self, id=id, size=size, data=None)
        self.data = ADTLChunk.Contents()
        start = 0
        while start < len(data)-1:
            subid = bytes.decode(data[start:start+4])
            if len(subid):
                sizesubid = int.from_bytes(data[start+4:start+8], byteorder="little")
                cueID = bytes.decode(data[start + 8:start + 12])
            if subid in self.Contents.__annotations__:
                exec("%s=%s" % ('self.data.'+subid, "ADTLsubChunk1(subid, sizesubid, cueID, data[start+12:start+sizesubid+12])"))
            else:
                # unrecognizedChunk = Chunk(subid, sizesubid, cueID, data[start+12:start+sizesubid+12])
                self.data.unrecognized = ADTLsubChunk1(subid, sizesubid, cueID, data[start+12:start+sizesubid+12])
                # print(unrecognizedChunk)
            start = start + sizesubid + 8

        if len(data) > 4 + start:
            self.data.unrecognized = ADTLsubChunk1(bytes.decode(data[start:start + 4]), len(data)-8, "", data[start + 4:])

    def __repr__(self):
        return Chunk.__repr__(self) + "\n" + str(self.data)
    pass

class LISTChunk(Chunk):
    class Contents:
        INFO: INFOChunk
        adtl: ADTLChunk

        def __repr__(self):
            list = ""
            try:
                list += str(self.INFO)
            except AttributeError:
                pass
            try:
                list += str(self.adtl)
            finally:
                return list
        pass

    data: Contents

    def __init__(self, id: str, size: int, data: [bytes, int]):
        Chunk.__init__(self=self, id=id, size=size, data=None)
        self.data = LISTChunk.Contents()
        start = 0
        while start < data.index(data[-1]):
            subid = bytes.decode(data[start:start+4])
            if subid == "INFO":
                self.data.INFO=INFOChunk(subid, len(data[start+4:start+size]), data[start+4:start+size])
            elif subid == "adtl":
                self.data.adtl = ADTLChunk(subid, len(data[start + 4:start + size]), data[start + 4:start + size])
            else:
                unrecognizedChunk = Chunk(subid, len(data[start+4:start+size]), data[start+4:start+size])
                print(unrecognizedChunk)     # niezapisywany bo nie ma takiego i to na 100% błąd
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
    class Contents:
        TPE1: INFOsubChunk  #wykonawca
        COMM: INFOsubChunk  #tytuł utworu
        TIT2: INFOsubChunk  #tytuł albumu
        TDRC: INFOsubChunk  # gatunek
        TALB: INFOsubChunk  #komentarze
        TRCK: INFOsubChunk  #oprogramowanie
        TCON: INFOsubChunk  #oprogramowanie
        unrecognized: INFOsubChunk

        def __repr__(self):
            list = ""
            try:
                list += str(self.TPE1)
            except AttributeError:
                pass
            try:
                list += str(self.COMM)
            except AttributeError:
                pass
            try:
                list += str(self.TIT2)
            except AttributeError:
                pass
            try:
                list += str(self.TDRC)
            except AttributeError:
                pass
            try:
                list += str(self.TALB)
            except AttributeError:
                pass
            try:
                list += str(self.TRCK)
            except AttributeError:
                pass
            try:
                list += str(self.TCON)
            except AttributeError:
                pass
            try:
                list += str(self.unrecognized)
            finally:
                return list
    data: Contents

    def __init__(self, id: str, size: int, data: list):
        Chunk.__init__(self=self, id=id, size=size, data=None)
        self.data = ID3Chunk.Contents()
        start = 0
        while len(data) > 45:
            subid = bytes.decode(data[start:start+4])
            if subid in self.Contents.__annotations__:
                for byte in data[1:-3]:
                    # print(bytes.decode(data[data.index(byte)+5:data.index(byte)+9]))
                    if bytes.decode(data[data.index(byte)+5:data.index(byte)+9]) in self.Contents.__annotations__:
                        exec("%s=%s" % ('self.data.' + subid, "INFOsubChunk(subid,data.index(byte)+5-4, data[start+4:start+data.index(byte)+5])"))
                        data = data[data.index(byte) + 5:]
                        break
            else:
                unrecognizedChunk = Chunk(subid, size-4, data[start+4:start+size])
                print(unrecognizedChunk)

        if len(data) > 4 + start:
            self.data.unrecognized = INFOsubChunk(bytes.decode(data[start:start+4]), len(data)-8, data[start+4:])

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
            if subid == "ID3":
                self.data.ID3=ID3Chunk(subid, size - 10, data[start+10:start+size])
            else:
                unrecognizedChunk = Chunk(subid, size-10, data[start+10:start+size])  # niezapisywany bo nie ma takiego
                print(unrecognizedChunk)                                              # i to na 100% błąd
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
    elif id == "fact":
        data = [f.read(size)]
        factChunk = RIFFHeader(id, size, data[0])
        print(factChunk)
    elif id == "data":
        if fmtChunk.data.bits_per_sample >= 8:
            sample_len = int(fmtChunk.data.bits_per_sample / 8)
        else:
            sample_len = int(fmtChunk.data.bits_per_sample/4)

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
        # print(dataChunk)
    else:
        data = [f.read(size)]
        unrecognizedChunk = Chunk(id, size, data)
        print(unrecognizedChunk)
        print(unrecognizedChunk.data)

# File = WavFile(riffChunk, fmtChunk, dataChunk, listChunk)

samples = dataChunk.data.samples
#TODO downsampling danych przy rysowaniu wykresów dla usprawnienia obliczenia i czasu rysowania
# plt.plot(range(0, 100, 1), samples[0:100])
# plt.plot(range(0, len(samples[0])), samples[0])
# plt.show(block=True)
# np.seterr(divide='ignore')  #TODO przy logarytmie wewnatrz tworzenia spektrogramu pojawia sie warning, tymczasowe
# plt.specgram(x=samples, Fs=fmtChunk.data.sample_rate, scale="dB")
# plt.yscale("symlog")
# plt.ylabel("Frequency [Hz]")
# plt.xlabel("Time [s]")
# plt.show(block=True)


