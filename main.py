from matplotlib import pyplot as plt
import scipy.fft
import struct
import audioop
import numpy as np
import os
import math

from wav_chunks import *

# Optional, index, tab, unrecognizedChunk = {}, 1, [], []

size = 0
sample_len = 0
f = open(file="data/gos_copy2.wav", mode="rb")

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


def display_information(riffChunk: RIFFHeader, dataChunk: DataChunk, fmtChunk: FmtChunk, optional: dict):
    print("Pomyślnie wczytano plik")
    print(riffChunk)
    print(fmtChunk)
    print(f"\tChunk ID: {dataChunk.id} Chunk size: {dataChunk.size}\n")
    try:
        if 'LIST' in Optional.values():
            print(listChunk)
    except Exception:
        pass
    try:
        if 'id3 ' in Optional.values():
            print(id3Chunk)
    except Exception:
        pass
    try:
        if 'fact' in Optional.values():
            print(factChunk)
    except Exception:
        pass
    try:
        if 'cue ' in Optional.values():
            print(cueChunk)
    except Exception:
        pass
    try:
        for unrecognized in unrecognizedChunk:
            try:
                print("\tNierozpoznany:")
                print(f"\tChunk ID: {unrecognized.id} Chunk size: {unrecognized.size}")
                print(f"\t\tData: {unrecognized.data}")
            except Exception:
                pass
    except Exception:
        pass



def normalize_samples(samples: list, fmtChunk: FmtChunk):
    samples = np.array(samples)

    np.seterr(divide='ignore')

    if fmtChunk.data.bits_per_sample >= 8:
        sample_len = int(fmtChunk.data.bits_per_sample / 8)
    else:
        sample_len = int(fmtChunk.data.bits_per_sample / 4)

    if fmtChunk.data.audio_format == 1 or fmtChunk.data.audio_format == 2:
        if sample_len == 1:
            return samples/(2**(fmtChunk.data.bits_per_sample-1)) - 1
        else:
            return samples/(2**(fmtChunk.data.bits_per_sample-1))
    if fmtChunk.data.audio_format == 3:
        return samples
    if fmtChunk.data.audio_format == 6 or fmtChunk.data.audio_format == 7:
        return samples/(2**(fmtChunk.data.bits_per_sample-1))


def display_waveform(dataChunk: DataChunk, fmtChunk: FmtChunk, lower: int = None, upper: int = None):
    plt.close()

    channels = np.array(dataChunk.data.samples)


    if lower is None or upper is None:
        lower = 0
        upper = len(channels[0])
    elif lower < 0 or upper < 0 or lower > len(channels[0]) or upper > len(channels[0]):
        lower = 0
        upper = len(channels[0])


    time_axis = (np.array(range(0, len(channels[0])))/fmtChunk.data.sample_rate)[lower:upper]
    figure, axes = plt.subplots(len(channels), 1, sharex=False, sharey=True)

    if fmtChunk.data.num_channels == 1:
        channel = channels[0][lower:upper]
        channel = normalize_samples(channel, fmtChunk)
        axes.plot(time_axis, channel)
        axes.set_ylabel("Znormalizowana amplituda")
        axes.set_xlabel("Czas [s]")
    else:
        for channel_index, channel in enumerate(channels):
            channel = channel[lower:upper]
            channel = normalize_samples(channel, fmtChunk)
            axes[channel_index].plot(time_axis, channel)
            axes[channel_index].set_title(f"Kanał {channel_index+1}")
            axes[channel_index].set_ylabel("Znormalizowana amplituda")
            axes[channel_index].set_xlabel("Czas [s]")
    plt.suptitle("Przebieg wybranego fragmentu sygnału wewnątrz pliku")
    plt.tight_layout()
    plt.show(block=True)


def display_spectrogram(dataChunk: DataChunk, fmtChunk: FmtChunk, lower, upper):
    plt.close()
    channels = np.array(dataChunk.data.samples)

    if lower is None or upper is None:
        lower = 0
        upper = len(channels[0])
    elif lower < 0 or upper < 0 or lower > len(channels[0]) or upper > len(channels[0]):
        lower = 0
        upper = len(channels[0])

    time_axis = (np.array(range(0, len(channels[0])))/fmtChunk.data.sample_rate)[lower:upper]
    figure, axes = plt.subplots(len(channels), 1, sharex=False, sharey=True)


    if fmtChunk.data.num_channels == 1:
        channel = channels[0]
        channel = channel[lower:upper]
        channel = normalize_samples(channel, fmtChunk)
        axes.specgram(x=channel, Fs=fmtChunk.data.sample_rate, scale="dB")
        axes.set_yscale("symlog")
        axes.set_ylabel("Częstotliwość [Hz]")
        axes.set_xlabel("Czas [s]")
    else:
        for channel_index, channel in enumerate(channels):
            channel = channel[lower:upper]
            channel = normalize_samples(channel, fmtChunk)
            axes[channel_index].specgram(x=channel, Fs=fmtChunk.data.sample_rate, scale="dB")
            axes[channel_index].set_yscale("symlog")
            axes[channel_index].set_title(f"Kanał {channel_index+1}")
            axes[channel_index].set_ylabel("Częstotliwość [Hz]")
            axes[channel_index].set_xlabel("Czas [s]")
    plt.suptitle("Spektrogram wybranego fragmentu sygnału wewnątrz pliku")
    plt.tight_layout()
    plt.show(block=True)


def display_amplitude_spectrum(dataChunk: DataChunk, fmtChunk: FmtChunk, lower: int = None, upper: int = None):
    plt.close()
    channels = np.array(dataChunk.data.samples)

    if lower is None or upper is None:
        lower = 0
        upper = len(channels[0])
    elif lower < 0 or upper < 0 or lower > len(channels[0]) or upper > len(channels[0]):
        lower = 0
        upper = len(channels[0])

    time_axis = (np.array(range(0, len(channels[0])))/fmtChunk.data.sample_rate)[lower:upper]
    figure, axes = plt.subplots(len(channels), 1, sharex=False, sharey=True)

    if fmtChunk.data.num_channels == 1:
        channel = channels[0][lower:upper]
        channel = normalize_samples(channel, fmtChunk)
        spectrum = scipy.fft.rfft(channel)
        frequencies = scipy.fft.rfftfreq(len(channel), 1 / fmtChunk.data.sample_rate)
        axes.plot(frequencies, np.abs(spectrum))
        axes.set_xscale("symlog")
        axes.set_ylabel("Amplituda")
        axes.set_xlabel("Częstotliwość [Hz]")
    else:
        for channel_index, channel in enumerate(channels):
            channel = channel[lower:upper]
            #channel = normalize_samples(channel, fmtChunk)
            spectrum = scipy.fft.rfft(channel)
            frequencies = scipy.fft.rfftfreq(len(channel), 1/fmtChunk.data.sample_rate)
            axes[channel_index].plot(frequencies, np.abs(spectrum))
            axes[channel_index].set_xscale("symlog")
            axes[channel_index].set_title(f"Kanał {channel_index+1}")
            axes[channel_index].set_ylabel("Amplituda")
            axes[channel_index].set_xlabel("Częstotliwość [Hz]")
    plt.suptitle("Widmo amplitudowe wybranego fragmentu sygnału wewnątrz pliku")
    plt.tight_layout()
    plt.show(block=True)


def display_phase_spectrum(dataChunk: DataChunk, fmtChunk: FmtChunk, lower: int = None, upper: int = None):
    plt.close()
    channels = np.array(dataChunk.data.samples)

    if lower is None or upper is None:
        lower = 0
        upper = len(channels[0])
    elif lower < 0 or upper < 0 or lower > len(channels[0]) or upper > len(channels[0]):
        lower = 0
        upper = len(channels[0])

    time_axis = (np.array(range(0, len(channels[0])))/fmtChunk.data.sample_rate)[lower:upper]
    figure, axes = plt.subplots(len(channels), 1, sharex=False, sharey=True)

    if fmtChunk.data.num_channels == 1:
        channel = channels[0][lower:upper]
        normalize_samples(channel, fmtChunk)
        spectrum = scipy.fft.rfft(channel)
        threshold = max(abs(spectrum)) / 100
        for i, x in enumerate(spectrum):
            if x < threshold:
                spectrum[i] = 0
        frequencies = scipy.fft.rfftfreq(len(channel), 1 / fmtChunk.data.sample_rate)
        axes.plot(frequencies, np.angle(spectrum))
        axes.set_xscale("symlog")
        axes.set_ylabel("Przesunięcie fazowe [rad]")
        axes.set_xlabel("Częstotliwość [Hz]")
    else:
        for channel_index, channel in enumerate(channels):
            channel = channel[lower:upper]
            channel = normalize_samples(channel, fmtChunk)
            spectrum = scipy.fft.rfft(channel)
            threshold = max(abs(spectrum))/100
            for i, x in enumerate(spectrum):
                if x < threshold:
                    spectrum[i] = 0
            frequencies = scipy.fft.rfftfreq(len(channel), 1/fmtChunk.data.sample_rate)
            axes[channel_index].plot(frequencies, np.angle(spectrum))
            axes[channel_index].set_xscale("symlog")
            axes[channel_index].set_title(f"Kanał {channel_index+1}")
            axes[channel_index].set_ylabel("Przesunięcie fazowe [rad]")
            axes[channel_index].set_xlabel("Częstotliwość [Hz]")
    plt.suptitle("Widmo fazowe wybranego fragmentu sygnału wewnątrz pliku")
    plt.tight_layout()
    plt.show(block=True)


display_information(riffChunk, dataChunk, fmtChunk, Optional)

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
