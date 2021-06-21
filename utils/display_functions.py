import scipy.fft

from utils.wav_chunks import *


def display_information(riff_chunk: RIFFHeader, data_chunk: DataChunk, fmt_chunk: FmtChunk, optional: dict,
                        list_chunk: LISTChunk = None, id3_chunk: ID3Chunk = None, fact_chunk: factChunk = None,
                        cue_chunk: CueChunk = None):
    print("Pomyślnie wczytano plik")
    print(riff_chunk)
    print(fmt_chunk)
    print(f"\tChunk ID: {data_chunk.id} Chunk size: {data_chunk.size}\n")
    try:
        if 'LIST' in optional.values() and list_chunk is not None:
            print(cue_chunk)
    except Exception:
        pass
    try:
        if 'id3 ' in optional.values() and id3_chunk is not None:
            print(id3_chunk)
    except Exception:
        pass
    try:
        if 'fact' in optional.values() and fact_chunk is not None:
            print(fact_chunk)

    except Exception:
        pass
    try:
        if 'cue ' in optional.values() and cue_chunk is not None:
            print(cue_chunk)
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
