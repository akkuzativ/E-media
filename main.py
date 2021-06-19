import rsa

from display_functions import *
from rsa_tools import rsa_lib_wrapper, encryption_utils

# Optional, index, tab, unrecognizedChunk = {}, 1, [], []

size = 0
sample_len = 0

listChunk = None
id3Chunk = None
factChunk = None
cueChunk = None

decrypt_file_contents = True
encryption_data_file_name = "encryption_data.yaml"
f = open(file="nowy.wav", mode="rb")

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
        data = f.read(size)
        if decrypt_file_contents:
            encryption_data = encryption_utils.read_rsa_data_from_file(encryption_data_file_name)
            data = rsa_lib_wrapper.decrypt_ebc(data, private_key=rsa.PrivateKey(*encryption_data))
        data = DataChunk.Contents.bytes_to_channels(fmtChunk, data, len(data))
        dataChunk = DataChunk(id, len(data), data)
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


(pub, priv) = rsa.key.newkeys(128)
samples_as_bytes = DataChunk.Contents.channels_to_bytes(fmtChunk, dataChunk.data)
encrypted_samples = rsa_lib_wrapper.encrypt_ebc(samples_as_bytes, public_key=pub)

encryption_utils.write_rsa_data_to_file(encryption_data_file_name, rsa_lib_wrapper.private_key_to_rsa_data(priv))

dataChunk = DataChunk(dataChunk.id, len(encrypted_samples), DataChunk.Contents.bytes_to_channels(fmtChunk, encrypted_samples, dataChunk.size))

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
