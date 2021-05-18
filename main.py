from matplotlib import pyplot as plt
import scipy.fft
import struct
import audioop
import numpy as np
import os


class Chunk:
    def __init__(self, id: str, size: int, data=None):
        self.id = id
        self.size = size
        self.data = data

    def __repr__(self):
        return str(self.id) + " - " + str(self.size)  # + " - " + str(self.data)

    def __str__(self):
        return f"Chunk ID: {self.id} Chunk size: {self.size}"

    pass

    def write(self, file):
        try:
            file.write(self.id.encode(encoding='utf-8'))
            file.write(self.size.to_bytes(4, byteorder="little", signed=True))
        except Exception:
            pass


class RIFFHeader(Chunk):
    class Contents:
        format: str

        def __init__(self, data: list):
            self.format = data[0]

        def __repr__(self):
            return str(self.format)

        pass

        def write(self, file):
            try:
                file.write(self.format.encode(encoding='utf-8'))
            except Exception:
                pass

    data: Contents

    def __init__(self, id: str, size: int, data: list):
        Chunk.__init__(self=self, id=id, size=size, data=None)
        self.data = RIFFHeader.Contents(data)

    def __repr__(self):
        return Chunk.__repr__(self) + "\n" + str(self.data)

    pass

    def write(self, file=None):
        Chunk.write(self, file)
        self.data.write(file)


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
            list = str(self.audio_format) + " - " + str(self.num_channels) + " - " + str(self.sample_rate) + \
                   " - " + str(self.byte_rate) + " - " + str(self.block_align) + " - " + str(self.bits_per_sample)
            if data.index(data[-1]) > 5:
                list += " - " + str(self.num_extra_format_bytes) + " - " + str(self.extra_format_bytes)
            return list

        def __str__(self):
            ret = f"Audio format: {self.audio_format}"
            ret += f"\nNumber of channels: {self.num_channels}"
            ret += f"\nSample rate: {self.sample_rate}"
            ret += f"\nByte rate: {self.byte_rate}"
            ret += f"\nBlock align: {self.block_align}"
            ret += f"\nBits per sample: {self.bits_per_sample}"
            if data.index(data[-1]) > 5:
                ret += f"\nNumber of extra format bytes {self.num_extra_format_bytes}"
                ret += f"\nExtra format bytes {self.extra_format_bytes}"
            return ret

        pass

        def write(self, file):
            try:
                file.write(self.audio_format.to_bytes(2, byteorder="little", signed=True))
                file.write(self.num_channels.to_bytes(2, byteorder="little", signed=True))
                file.write(self.sample_rate.to_bytes(4, byteorder="little", signed=True))
                file.write(self.byte_rate.to_bytes(4, byteorder="little", signed=True))
                file.write(self.block_align.to_bytes(2, byteorder="little", signed=True))
                file.write(self.bits_per_sample.to_bytes(2, byteorder="little", signed=True))
                try:
                    file.write(self.num_extra_format_bytes.to_bytes(2, byteorder="little", signed=True))
                    file.write(
                        self.extra_format_bytes.to_bytes(self.data.num_extra_format_bytes + 1, byteorder="little",
                                                         signed=True))
                    file.seek(-1, np.os.SEEK_END)
                    file.truncate()
                except Exception:
                    pass
            except Exception:
                pass

    data: Contents

    def __init__(self, id: str, size: int, data: list):
        Chunk.__init__(self=self, id=id, size=size, data=None)
        self.data = FmtChunk.Contents(data)

    def __repr__(self):
        return Chunk.__repr__(self) + "\n" + str(self.data)

    def __str__(self):
        return Chunk.__repr__(self) + "\n" + str(self.data)


    pass

    def write(self, file=None):
        Chunk.write(self, file)
        self.data.write(file)

class factChunk(Chunk):
    class Contents:
        data: int

        def __init__(self, data: int):
            self.data = data

        def __repr__(self):
            return str(self.data)

        pass

        def write(self, file):
            try:
                file.write(self.data.to_bytes(4, byteorder="little", signed=True))
            except Exception:
                pass
    data: Contents

    def __init__(self, id: str, size: int, data: list):
        Chunk.__init__(self=self, id=id, size=size, data=None)
        self.data = factChunk.Contents(int.from_bytes(data[0], byteorder="little"))

    def __repr__(self):
        return Chunk.__repr__(self) + " - " + str(self.data) + " - "

    pass

    def write(self, file):
        Chunk.write(self, file)
        self.data.write(file)

class INFOsubChunk(Chunk):
    class Contents:
        data: str

        def __init__(self, data: str):
            self.data = data

        def __repr__(self):
            return str(self.data)

        pass

        def write(self, file):
            try:
                file.write(self.data.encode(encoding="utf-8"))
            except Exception:
                pass
    data: Contents

    def __init__(self, id: str, size: int, data: list):
        Chunk.__init__(self=self, id=id, size=size, data=None)
        self.data = INFOsubChunk.Contents(bytes.decode(data[:]))

    def __repr__(self):
        return Chunk.__repr__(self) + " - " + str(self.data) + " - "

    pass

    def write(self, file):
        Chunk.write(self, file)
        self.data.write(file)


class INFOChunk(Chunk):
    class Contents:
        IART: INFOsubChunk          # wykonawca
        INAM: INFOsubChunk          # tytuł utworu
        IPRD: INFOsubChunk          # tytuł albumu
        ICRD: INFOsubChunk          # data wydania
        IGNR: INFOsubChunk          # gatunek
        ICMT: INFOsubChunk          # komentarze
        ITRK: INFOsubChunk          # komentarze
        ISFT: INFOsubChunk          # oprogramowanie
        unrecognized: INFOsubChunk  # nierozpoznany

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

        def write(self, file):
            try:
                if 'INAM' in tab:
                    self.INAM.write(file)
            except AttributeError:
                pass
            try:
                if 'IPRD' in tab:
                    self.IPRD.write(file)
            except AttributeError:
                pass
            try:
                if 'IART' in tab:
                    self.IART.write(file)
            except AttributeError:
                pass
            try:
                if 'ICMT' in tab:
                    self.ICMT.write(file)
            except AttributeError:
                pass
            try:
                if 'ICRD' in tab:
                   self.ICRD.write(file)
            except AttributeError:
                pass
            try:
                if 'IGNR' in tab:
                    self.IGNR.write(file)
            except AttributeError:
                pass
            try:
                if 'ITRK' in tab:
                   self.ITRK.write(file)
            except AttributeError:
                pass
            try:
                if 'ISFT' in tab:
                    self.ISFT.write(file)
            except AttributeError:
                pass
            try:
                file.write(self.unrecognized.id.encode(encoding='utf-8'))
                file.write(self.unrecognized.data.data.encode(encoding='utf-8'))
            except AttributeError:
                pass

    data: Contents

    def __init__(self, id: str, size: int, data: list):
        global index
        Chunk.__init__(self=self, id=id, size=size, data=None)
        self.data = INFOChunk.Contents()
        start = 0
        while start < len(data) - 1:
            subid = bytes.decode(data[start:start + 4])
            if len(subid):
                sizesubid = int.from_bytes(data[start + 4:start + 8], byteorder="little")
            if subid in self.Contents.__annotations__:
                exec(
                    "%s=%s" % ('self.data.' + subid, "INFOsubChunk(subid, sizesubid, data[start+8:start+sizesubid+8])"))
                exec("%s" % ("Optional.update({index: subid})"))
                index += 1
            else:
                self.data.unrecognized = INFOsubChunk(subid, sizesubid, data[start + 8:start + sizesubid + 8])
                Optional.update({index: self.id + self.data.unrecognized.id})
                index += 1
            start = start + sizesubid + 8

        if len(data) > 4 + start:
            self.data.unrecognized = INFOsubChunk(bytes.decode(data[start:start + 4]), len(data[start + 4:]),
                                                  data[start + 4:])
            Optional.update({index: self.id + self.data.unrecognized.id})
            index += 1

    def __repr__(self):
        return Chunk.__repr__(self) + "\n" + str(self.data)

    pass

    def write(self, file):
        file.write(self.id.encode(encoding='utf-8'))
        self.data.write(file)


class ADTLsubChunk(Chunk):
    class Contents:
        cueID: str
        sample: str
        purpouse: str
        country: str
        lang: str
        dial: str
        code: str
        data: str

        def __init__(self, cueID: str, data: str, sample: str = None, purpouse: str = None, country: str = None,
                     lang: str = None, dial: str = None, code: str = None):
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

        def write(self, file, id):
            try:
                file.write(self.cueID.encode(encoding="utf-8"))
                if id == "itxt":
                    file.write(self.cueID.encode(encoding="utf-8"))
                    file.write(self.sample.encode(encoding="utf-8"))
                    file.write(self.purpouse.encode(encoding="utf-8"))
                    file.write(self.country.encode(encoding="utf-8"))
                    file.write(self.lang.encode(encoding="utf-8"))
                    file.write(self.dial.encode(encoding="utf-8"))
                    file.write(self.code.encode(encoding="utf-8"))

                file.write(self.data.encode(encoding="utf-8"))
            except Exception:
                pass

    data: Contents

    def __init__(self, id: str, size: int, cueID: str, data: list):
        Chunk.__init__(self=self, id=id, size=size, data=None)
        if id != "ltxt":
            self.data = ADTLsubChunk.Contents(cueID, bytes.decode(data[:]))
        else:
            self.data = ADTLsubChunk.Contents(cueID, bytes.decode(data[16:]), bytes.decode(data[:4]),
                                              bytes.decode(data[4:8]), bytes.decode(data[8:10]),
                                              bytes.decode(data[10:12]), bytes.decode(data[12:14]),
                                              bytes.decode(data[14:16]))

    def __repr__(self):
        return Chunk.__repr__(self) + " - " + str(self.data)

    pass

    def write(self, file):
        Chunk.write(self, file)
        self.data.write(file, self.id)


class ADTLChunk(Chunk):
    class Contents:
        labl: ADTLsubChunk
        note: ADTLsubChunk
        ltxt: ADTLsubChunk
        unrecognized: ADTLsubChunk

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

        def write(self, file):
            try:
                if 'labl' in tab:
                    self.labl.write(file)
            except AttributeError:
                pass
            try:
                if 'note' in tab:
                    self.note.write(file)
            except AttributeError:
                pass
            try:
                if 'ltxt' in tab:
                    self.ltxt.write(file)
            except AttributeError:
                pass
            try:
                file.write(self.unrecognized.id.encode(encoding='utf-8'))
                file.write(self.unrecognized.data.data.encode(encoding='utf-8'))
            except AttributeError:
                pass

    data: Contents

    def __init__(self, id: str, size: int, data: list):
        global index
        Chunk.__init__(self=self, id=id, size=size, data=None)
        self.data = ADTLChunk.Contents()
        start = 0
        while start < len(data) - 1:
            subid = bytes.decode(data[start:start + 4])
            if len(subid):
                sizesubid = int.from_bytes(data[start + 4:start + 8], byteorder="little")
                cueID = bytes.decode(data[start + 8:start + 12])
            if subid in self.Contents.__annotations__:
                exec("%s=%s" % (
                    'self.data.' + subid, "ADTLsubChunk(subid, sizesubid, cueID, data[start+12:start+sizesubid+12])"))
                exec("%s" % ("Optional.update({index: subid})"))
                index += 1
            else:
                self.data.unrecognized = ADTLsubChunk(subid, sizesubid, cueID, data[start + 12:start + sizesubid + 12])
                Optional.update({index: self.id + self.data.unrecognized.id})
                index += 1
            start = start + sizesubid + 8

        if len(data) > 4 + start:
            self.data.unrecognized = ADTLsubChunk(bytes.decode(data[start:start + 4]), len(data[start + 4:]), "",
                                                  data[start + 4:])
            Optional.update({index: self.id + self.data.unrecognized.id})
            index += 1

    def __repr__(self):
        return Chunk.__repr__(self) + "\n" + str(self.data)

    pass

    def write(self, file):
        file.write(self.id.encode(encoding='utf-8'))
        self.data.write(file)


class LISTChunk(Chunk):
    class Contents:
        INFO: INFOChunk
        adtl: ADTLChunk
        unrecognized: INFOsubChunk

        def __repr__(self):
            list = ""
            try:
                list += str(self.INFO)
            except AttributeError:
                pass
            try:
                list += str(self.adtl)
            except AttributeError:
                pass
            try:
                list += str(self.unrecognized)
            finally:
                return list

        pass

        def write(self, file):
            try:
                if 'INFO' in tab:
                    self.INFO.write(file)
            except Exception:
                pass
            try:
                if 'adtl' in tab:
                    self.adtl.write(file)
            except Exception:
                pass
            try:
                file.write(self.unrecognized.id.encode(encoding='utf-8'))
                file.write(self.unrecognized.data.data.encode(encoding='utf-8'))
            except Exception:
                pass

    data: Contents

    def __init__(self, id: str, size: int, data: [bytes, int]):
        global index
        Chunk.__init__(self=self, id=id, size=size, data=None)
        self.data = LISTChunk.Contents()
        start = 0
        while start < data.index(data[-1]):
            subid = bytes.decode(data[start:start + 4])
            if subid == "INFO":
                self.data.INFO = INFOChunk(subid, len(data[start + 4:start + size]), data[start + 4:start + size])
                Optional.update({index: self.data.INFO.id})
                index += 1
            elif subid == "adtl":
                self.data.adtl = ADTLChunk(subid, len(data[start + 4:start + size]), data[start + 4:start + size])
                Optional.update({index: self.data.adtl.id})
                index += 1
            else:
                self.data.unrecognized = INFOsubChunk(subid, len(data[start + 4:start + size]),
                                                      data[start + 4:start + size])
                Optional.update({index: self.id + self.data.unrecognized.id})
                index += 1
            start = start + size

    def __repr__(self):
        return Chunk.__repr__(self) + "\n" + str(self.data)

    pass

    def write(self, file):
        Chunk.write(self, file)
        self.data.write(file)


class DataChunk(Chunk):
    class Contents:
        samples: list

        def __init__(self, data: list):
            self.samples = data

        def __repr__(self):
            return str(self.samples)

        pass

        def write(self, file, fmtChunk: FmtChunk):

            def sample_conversion(sample, fmtChunk: FmtChunk):
                if fmtChunk.data.bits_per_sample >= 8:
                    sample_len = int(fmtChunk.data.bits_per_sample / 8)
                else:
                    sample_len = int(fmtChunk.data.bits_per_sample / 4)
                if fmtChunk.data.audio_format == 1:
                    if sample_len == 1:
                        return int.to_bytes(sample, byteorder="little", signed=False, length=sample_len)
                    else:
                        return int.to_bytes(sample, byteorder="little", signed=True, length=sample_len)
                elif fmtChunk.data.audio_format == 3:
                    if sample_len == 4:
                        return bytearray(struct.pack("f", sample))
                    else:
                        return bytearray(struct.pack("d", sample))
                elif fmtChunk.data.audio_format == 6:
                    return audioop.lin2alaw(sample, sample_len)
                elif fmtChunk.data.audio_format == 7:
                    return audioop.lin2ulaw(sample, sample_len)

            for channel in self.samples:
                for sample in channel:
                    bytes_sample = sample_conversion(sample, fmtChunk)
                    file.write(bytes_sample)


    data: Contents

    def __init__(self, id: str, size: int, data: list):
        Chunk.__init__(self=self, id=id, size=size, data=None)
        self.data = DataChunk.Contents(data)

    def __repr__(self):
        return Chunk.__repr__(self) + "\n" + str(self.data)

    pass

    def write(self, file, fmtChunk: FmtChunk):
        Chunk.write(self, file)
        self.data.write(file, fmtChunk)


class ID3Chunk(Chunk):
    class Contents:
        TPE1: INFOsubChunk          # wykonawca
        COMM: INFOsubChunk          # tytuł utworu
        TIT2: INFOsubChunk          # tytuł albumu
        TDRC: INFOsubChunk          # gatunek
        TALB: INFOsubChunk          # komentarze
        TRCK: INFOsubChunk          # oprogramowanie
        TCON: INFOsubChunk          # oprogramowanie
        unrecognized: INFOsubChunk  # nierozpoznany

        def __repr__(self):
            list = ""
            try:
                list += str(self.TPE1)
            except AttributeError:
                pass
            try:
                list += str(self.TIT2)
            except AttributeError:
                pass
            try:
                list += str(self.COMM)
            except AttributeError:
                pass
            try:
                list += str(self.TALB)
            except AttributeError:
                pass
            try:
                list += str(self.TDRC)
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

        def write(self, file):
            try:
                if 'TPE1' in tab:
                    self.TPE1.write(file)
            except Exception:
                pass
            try:
                if 'TIT2' in tab:
                    self.TIT2.write(file)
            except:
                pass
            try:
                if 'COMM' in tab:
                    self.COMM.write(file)
            except:
                pass
            try:
                if 'TALB' in tab:
                    self.TALB.write(file)
            except:
                pass
            try:
                if 'TDRC' in tab:
                    self.TDRC.write(file)
            except:
                pass
            try:
                if 'TRCK' in tab:
                    self.TRCK.write(file)
            except:
                pass
            try:
                if 'TCON' in tab:
                    self.TCON.write(file)
            except:
                pass
            try:
                file.write(self.unrecognized.id.encode(encoding='utf-8'))
                file.write(self.unrecognized.data.data.encode(encoding='utf-8'))
            except Exception:
                pass

    data: Contents

    def __init__(self, id: str, size: int, data: list):
        global index
        Chunk.__init__(self=self, id=id, size=size, data=None)
        self.data = ID3Chunk.Contents()
        start = 0
        while len(data) > 45:
            subid = bytes.decode(data[start:start + 4])
            if subid in self.Contents.__annotations__:
                for byte in data[1:-3]:
                    if bytes.decode(data[data.index(byte) + 5:data.index(byte) + 9]) in self.Contents.__annotations__:
                        exec("%s=%s" % ('self.data.' + subid,
                                        "INFOsubChunk(subid,int.from_bytes(data[start + 4:start + 8], byteorder=\"little\"), data[start+8:start+data.index(byte)+5])"))
                        exec("%s" % ("Optional.update({index: subid})"))
                        index += 1
                        data = data[data.index(byte) + 5:]
                        break
            else:
                self.data.unrecognized = INFOsubChunk(subid, len(data[start + 4:]),
                                                      data[start + 4:])
                Optional.update({index: self.id + self.data.unrecognized.id})
                index += 1

        if len(data) > 4 + start:
            self.data.unrecognized = INFOsubChunk(bytes.decode(data[start:start + 4]), len(data[start + 4:]) - 8,
                                                  data[start + 4:])
            Optional.update({index: self.id + self.data.unrecognized.id})
            index += 1

    def __repr__(self):
        return Chunk.__repr__(self) + "\n" + str(self.data)

    pass

    def write(self, file):
        file.write(self.id.encode(encoding='utf-8'))
        file.write(self.size.to_bytes(7, byteorder='big', signed=True))
        self.data.write(file)


class id3Chunk(Chunk):
    class Contents:
        ID3: ID3Chunk
        unrecognized: INFOsubChunk

        def __repr__(self):
            list = ""
            try:
                list += str(self.ID3)
            except AttributeError:
                pass
            try:
                list += str(self.unrecognized)
            finally:
                return list

        pass

        def write(self, file):
            try:
                if 'ID3' in tab:
                    self.ID3.write(file)
            except Exception:
                pass
            try:
                file.write(self.unrecognized.id.encode(encoding='utf-8'))
                file.write(self.unrecognized.data.data.encode(encoding='utf-8'))
            except Exception:
                pass

    data: Contents

    def __init__(self, id: str, size: int, data: [bytes, int]):
        global index
        Chunk.__init__(self=self, id=id, size=size, data=None)
        self.data = id3Chunk.Contents()
        start = 0
        while start < data.index(data[-1]):
            subid = bytes.decode(data[start:start + 3])
            if subid == "ID3":
                subsize = int.from_bytes(data[start + 3:start + 10], byteorder="big")
                self.data.ID3 = ID3Chunk(subid, subsize, data[start + 10:start + size])
                Optional.update({index: self.data.ID3.id})
                index += 1
            else:
                self.data.unrecognized = INFOsubChunk(subid, len(data[start + 4:start + size]),
                                                      data[start + 4:start + size])
                Optional.update({index: self.id + self.data.unrecognized.id})
                index += 1

            start = start + size

    def __repr__(self):
        return Chunk.__repr__(self) + "\n" + str(self.data)

    pass

    def write(self, file):
        Chunk.write(self, file)
        self.data.write(file)


# class WavFile:
#     def __init__(self, a: RIFFHeader = None, b: FmtChunk = None, c: DataChunk = None, d: LISTChunk = None,
#                  e: list = None):
#         self.riff_descriptor = a
#         self.fmt = b
#         self.data = c
#         self.list = d
#         self.unrecognizedChunks = e
#
#     pass


size = 0
sample_len = 0
f = open(file="data/sine404-list2.wav", mode="rb")
Optional = {}
index = 1
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
        # print(riffChunk)
    elif id == "fmt ":
        data = [int.from_bytes(f.read(2), byteorder="little"), int.from_bytes(f.read(2), byteorder="little"),
                int.from_bytes(f.read(4), byteorder="little"), int.from_bytes(f.read(4), byteorder="little"),
                int.from_bytes(f.read(2), byteorder="little"), int.from_bytes(f.read(2), byteorder="little")]
        if size > 16:
            data.append(int.from_bytes(f.read(2), byteorder="little"))
            data.append(int.from_bytes(f.read(data[6]), byteorder="little"))
        fmtChunk = FmtChunk(id, size, data)
        # print(fmtChunk)
    elif id == "LIST":
        data = [f.read(size)]
        listChunk = LISTChunk(id, size, data[0])
        # print(listChunk)
        Optional.update({index: listChunk.id})
        index += 1
    elif id == "id3 ":
        data = [f.read(size)]
        id3Chunk = id3Chunk(id, size, data[0])
        # print(id3Chunk)
        Optional.update({index: id3Chunk.id})
        index += 1
    elif id == "fact":
        data = [f.read(size)]
        factChunk = factChunk(id, size, data)
        # print(factChunk)
        Optional.update({index: factChunk.id})
        index += 1
    elif id == "data":
        if fmtChunk.data.bits_per_sample >= 8:
            sample_len = int(fmtChunk.data.bits_per_sample / 8)
        else:
            sample_len = int(fmtChunk.data.bits_per_sample / 4)


        def sample_conversion(sample):
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

        raw_samples = f.read(size)
        samples = []

        # wykorzystanie klasycznego sposobu wczytywania - iteracje w pętli
        for i in range(int(size / sample_len)):
            samples.append(sample_conversion(raw_samples[i * sample_len:i * sample_len + sample_len]))

        channels = []
        for c in range(fmtChunk.data.num_channels):
            channels.append(samples[c::fmtChunk.data.num_channels])
        data = channels
        dataChunk = DataChunk(id, size, data)
        # print(dataChunk)
    else:
        data = [f.read(size)]
        unrecognizedChunk = Chunk(id, size, data)
        # print(unrecognizedChunk)

f.close()


def display_information(riffChunk: RIFFHeader, dataChunk: DataChunk, fmtChunk: FmtChunk, optional: dict):
    print("Pomyślnie wczytano plik")
    print(fmtChunk)


def display_waveform(dataChunk: DataChunk, fmtChunk: FmtChunk):
    plt.close()
    channels = np.array(dataChunk.data.samples)
    time_axis = np.array(range(0, len(channels[0])))/fmtChunk.data.sample_rate
    figure, axes = plt.subplots(len(channels), 1, sharex=False, sharey=True)
    if fmtChunk.data.num_channels == 1:
        channel = channels[0]
        channel = channel / (2 ** (fmtChunk.data.bits_per_sample - 1))
        axes.plot(time_axis, channel)
        axes.set_ylabel("Znormalizowana amplituda")
        axes.set_xlabel("Czas [s]")
    else:
        for channel_index, channel in enumerate(channels):
            if fmtChunk.data.audio_format == 1:
                channel = channel/(2**(fmtChunk.data.bits_per_sample-1))
            axes[channel_index].plot(time_axis, channel)
            axes[channel_index].set_title(f"Kanał {channel_index+1}")
            axes[channel_index].set_ylabel("Znormalizowana amplituda")
            axes[channel_index].set_xlabel("Czas [s]")
    plt.suptitle("Przebieg sygnału wewnątrz pliku")
    plt.tight_layout()
    plt.show(block=True)


def display_spectrogram(dataChunk: DataChunk, fmtChunk: FmtChunk):
    plt.close()
    channels = np.array(dataChunk.data.samples)
    time_axis = np.array(range(0, len(channels[0])))/fmtChunk.data.sample_rate
    figure, axes = plt.subplots(len(channels), 1, sharex=False, sharey=True)
    if fmtChunk.data.num_channels == 1:
        channel = channels[0]
        channel = channel / (2 ** (fmtChunk.data.bits_per_sample - 1))
        axes.specgram(x=channel, Fs=fmtChunk.data.sample_rate, scale="dB")
        axes.set_yscale("symlog")
        axes.set_ylabel("Częstotliwość [Hz]")
        axes.set_xlabel("Czas [s]")
    else:
        for channel_index, channel in enumerate(channels):
            if fmtChunk.data.audio_format == 1:
                channel = channel/(2**(fmtChunk.data.bits_per_sample-1))
            axes[channel_index].specgram(x=channel, Fs=fmtChunk.data.sample_rate, scale="dB")
            axes[channel_index].set_yscale("symlog")
            axes[channel_index].set_title(f"Kanał {channel_index+1}")
            axes[channel_index].set_ylabel("Częstotliwość [Hz]")
            axes[channel_index].set_xlabel("Czas [s]")
    plt.suptitle("Spektrogram sygnału wewnątrz pliku")
    plt.tight_layout()
    plt.show(block=True)


def display_amplitude_spectrum(dataChunk: DataChunk, fmtChunk: FmtChunk):
    plt.close()
    channels = np.array(dataChunk.data.samples)
    time_axis = np.array(range(0, len(channels[0])))/fmtChunk.data.sample_rate
    figure, axes = plt.subplots(len(channels), 1, sharex=False, sharey=True)
    if fmtChunk.data.num_channels == 1:
        channel = channels[0]
        channel = channel / (2 ** (fmtChunk.data.bits_per_sample - 1))
        axes.plot(time_axis, channel)
        amplitude = scipy.fft.rfft(channel)
        frequencies = scipy.fft.rfftfreq(len(channel), 1 / fmtChunk.data.sample_rate)
        axes.plot(frequencies, np.abs(amplitude))
        axes.set_xscale("symlog")
        axes.set_ylabel("Amplituda")
        axes.set_xlabel("Częstotliwość [Hz]")
    else:
        for channel_index, channel in enumerate(channels):
            if fmtChunk.data.audio_format == 1:
                channel = channel/(2**(fmtChunk.data.bits_per_sample-1))
            amplitude = scipy.fft.rfft(channel)
            frequencies = scipy.fft.rfftfreq(len(channel), 1/fmtChunk.data.sample_rate)
            axes[channel_index].plot(frequencies, np.abs(amplitude))
            axes[channel_index].set_xscale("symlog")
            axes[channel_index].set_title(f"Kanał {channel_index+1}")
            axes[channel_index].set_ylabel("Amplituda")
            axes[channel_index].set_xlabel("Częstotliwość [Hz]")
    plt.suptitle("Widmo amplitudowe sygnału wewnątrz pliku")
    plt.tight_layout()
    plt.show(block=True)


def display_phase_spectrum(dataChunk: DataChunk, fmtChunk: FmtChunk):
    pass


display_information(riffChunk, dataChunk, fmtChunk, Optional)
display_waveform(dataChunk, fmtChunk)
display_spectrogram(dataChunk, fmtChunk)
display_amplitude_spectrum(dataChunk, fmtChunk)

###
# zapis

print("Podaj indeksy które metadane zapisać do pliku, zakończ wybór wpisując literę:")
print("(Pamiętaj, żęby podać wszystkie chunki zawierające chunk, który chcesz zapisać):")
print(Optional)
tab = []
while True:
    try:
        elem = Optional.get(int(input()))
        tab.append(elem) if elem not in tab else None
    except Exception:
        break

file = open("nowy.wav", "wb")
riffChunk.write(file)
fmtChunk.write(file)
dataChunk.write(file, fmtChunk)
try:
    if 'LIST' in tab:
        listChunk.write(file)
except Exception:
    pass
try:
    if 'id3 ' in tab:
        id3Chunk.write(file)
except Exception:
    pass
try:
    if 'fact' in tab:
        factChunk.write(file)
except Exception:
    pass
file.close()
