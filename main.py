from matplotlib import pyplot as plt
import scipy.fft
import struct
import audioop
import numpy as np
import os
import math

from wav_chunks import *
from display_functions import *

# Optional, index, tab, unrecognizedChunk = {}, 1, [], []

size = 0
sample_len = 0

listChunk = None
id3Chunk = None
factChunk = None
cueChunk = None


f = open(file="data/sine440.wav", mode="rb")

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
    elif id == "fmt ":
        data = [int.from_bytes(f.read(2), byteorder="little"), int.from_bytes(f.read(2), byteorder="little"),
                int.from_bytes(f.read(4), byteorder="little"), int.from_bytes(f.read(4), byteorder="little"),
                int.from_bytes(f.read(2), byteorder="little"), int.from_bytes(f.read(2), byteorder="little")]
        if size > 16:
            data.append(int.from_bytes(f.read(2), byteorder="little"))
            data.append(int.from_bytes(f.read(data[6]), byteorder="little"))
        fmtChunk = FmtChunk(id, size, data)
    elif id == "LIST":
        data = [f.read(size)]
        listChunk = LISTChunk(id, size, data[0])
        Optional.update({index: listChunk.id})
        index += 1
    elif id == "id3 ":
        data = [f.read(size)]
        id3Chunk = id3Chunk(id, size, data[0])
        Optional.update({index: id3Chunk.id})
        index += 1
    elif id == "fact":
        data = [f.read(size)]
        factChunk = factChunk(id, size, data)
        Optional.update({index: factChunk.id})
        index += 1
    elif id == "cue ":
        data = [f.read(size)]
        cueChunk = CueChunk(id, size, data)
        Optional.update({index: cueChunk.id})
        index += 1
    elif id == "data":
        if int(fmtChunk.data.bits_per_sample / 8) > 0:
            sample_len = int(fmtChunk.data.bits_per_sample / 8)
        else:
            sample_len = 0

        adpcm_last_state = None

        raw_samples = f.read(size)
        samples = []

        if sample_len > 0:
            for i in range(int(size / sample_len)):
                sample = raw_samples[i * sample_len:i * sample_len + sample_len]
                if fmtChunk.data.audio_format == 1:
                    if sample_len == 1:
                        converted_sample = int.from_bytes(sample, byteorder="little", signed=False)
                    else:
                        converted_sample = int.from_bytes(sample, byteorder="little", signed=True)
                elif fmtChunk.data.audio_format == 3:
                    if sample_len == 4:
                        converted_sample = struct.unpack("f", sample)[0]
                    else:
                        converted_sample = struct.unpack("d", sample)[0]
                elif fmtChunk.data.audio_format == 6:
                    converted_sample = int.from_bytes(audioop.alaw2lin(sample, sample_len), byteorder="little", signed=True)
                elif fmtChunk.data.audio_format == 7:
                    converted_sample = int.from_bytes(audioop.ulaw2lin(sample, sample_len), byteorder="little", signed=True)
                else:
                    print("Format zapisu danych w pliku nie jest wspierany")
                    raise Exception
                samples.append(converted_sample)
        else:
            if fmtChunk.data.audio_format == 2:
                ret = audioop.adpcm2lin(raw_samples, fmtChunk.data.bits_per_sample, None)
                samples_lin = ret[0]
                for i in range(int(len(samples_lin)/8)):
                    sample = samples_lin[i*8:i*8+8]
                    converted_sample = int.from_bytes(sample, byteorder="little", signed=True)
                    samples.append(converted_sample)
            else:
                print("Format zapisu danych w pliku nie jest wspierany")
                raise Exception

        channels = []
        for c in range(fmtChunk.data.num_channels):
            channels.append(samples[c::fmtChunk.data.num_channels])
        data = channels
        dataChunk = DataChunk(id, size, data)
        # print(dataChunk)
    else:
        data = f.read(size)
        unrecognizedChunk.append(Chunk(id, size, data))

f.close()


display_information(riffChunk, dataChunk, fmtChunk, Optional, listChunk, id3Chunk, factChunk, cueChunk)

print("\n\nWybierz przedział próbek, z których zostanie narysowany przebieg oraz widma (najpierw dolny indeks, następnie górny, w przypadku nieprawidłowych indeksów wybrana zostanie całość)")
print(f"(min: 0 --- max: {len(dataChunk.data.samples[0])-1})")
print("Dolny indeks: ", end="")
try:
    lower = int(input())
except:
    lower = None
print("Górny indeks: ", end="")
try:
    upper = int(input())
except:
    upper = None


display_waveform(dataChunk, fmtChunk, lower, upper)
display_amplitude_spectrum(dataChunk, fmtChunk, lower, upper)
display_phase_spectrum(dataChunk, fmtChunk, lower, upper)
display_spectrogram(dataChunk, fmtChunk, lower, upper)

###
# zapis

print("\n\nPodaj indeksy które metadane zapisać do pliku, zakończ wybór wpisując literę:")
print("(Pamiętaj, żeby podać wszystkie chunki zawierające chunk, który chcesz zapisać):")
print(Optional)
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
try:
    if 'cue ' in tab:
        cueChunk.write(file)
except Exception:
    pass

if 'id3 ' in tab:
    file.write((0).to_bytes(1, byteorder="little", signed=True))
file.close()
