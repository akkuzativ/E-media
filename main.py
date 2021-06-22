import rsa
from utils.display_functions import *
from utils import rsa_lib_wrapper, encryption_utils, rsa_wrapper


## konfiguracja wykonania skryptu ##
skip_display = False
decrypt_file_contents_on_read = True
encrypt_file_contents_on_save = True
use_library_rsa = False
use_cbc = False
generate_new_keys = True
new_key_bit_len = 128
encryption_data_file_name = "encryption_data.yaml"

f = open(file="nowy.wav", mode="rb")
###################################


size = 0
sample_len = 0

listChunk = None
id3Chunk = None
factChunk = None
cueChunk = None

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
        raw_data = data
        if decrypt_file_contents_on_read:
            encryption_data = encryption_utils.read_rsa_data_from_file(encryption_data_file_name)
            try:
                if encryption_data.block_leftover_len is None: # to pole jest puste jesli wykorzystano szyfrowanie z biblioteki
                    if encryption_data.init_vector is None:
                        data = rsa_lib_wrapper.decrypt_ecb(data, private_key=rsa.PrivateKey(*encryption_data))
                    else:
                        data = rsa_lib_wrapper.decrypt_cbc(data, private_key=rsa.PrivateKey(*encryption_data),
                                                           init_vector=encryption_data.init_vector)
                    raw_data = data
                    data = DataChunk.Contents.bytes_to_channels(fmtChunk, data, len(data))
                    print("Pomyślnie odszyfrowano dane wewnątrz pliku.")
                else:
                    encryption_data = encryption_utils.read_rsa_data_from_file(encryption_data_file_name)
                    if encryption_data.init_vector is None:
                        data = rsa_wrapper.decrypt_ecb(data, encryption_data,
                                                       block_leftover_len=encryption_data.block_leftover_len)
                    else:
                        data = rsa_wrapper.decrypt_cbc(data, encryption_data,
                                                       init_vector=encryption_data.init_vector,
                                                       block_leftover_len=encryption_data.block_leftover_len)
                    raw_data = data
                    data = DataChunk.Contents.bytes_to_channels(fmtChunk, data, len(data))
                    print("Pomyślnie odszyfrowano dane wewnątrz pliku.")
            except Exception:
                print("Odszyfrowanie nie powiodło się. Wczytano dane w wersji niezmodyfikowanej.")
                data = DataChunk.Contents.bytes_to_channels(fmtChunk, data, size)
        else:
            data = DataChunk.Contents.bytes_to_channels(fmtChunk, data, size)
        dataChunk = DataChunk(id, len(data[0]), data)
    else:
        data = f.read(size)
        unrecognizedChunk.append(Chunk(id, size, data))

f.close()


display_information(riffChunk, dataChunk, fmtChunk, Optional, listChunk, id3Chunk, factChunk, cueChunk)

if not skip_display:
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


encrypted_samples = None


if encrypt_file_contents_on_save:
    if generate_new_keys:
        if use_library_rsa:
            encryption_data = rsa_lib_wrapper.private_key_to_rsa_data(rsa.newkeys(new_key_bit_len)[1])
        else:
            encryption_data = rsa_wrapper.new_keys(new_key_bit_len)
    else:
        encryption_data = encryption_utils.read_rsa_data_from_file(encryption_data_file_name)
        if not use_cbc:
            encryption_data.init_vector = None
        if use_library_rsa:
            encryption_data.block_leftover_len = None

    samples_as_bytes = DataChunk.Contents.channels_to_bytes(fmtChunk, dataChunk.data)

    if use_cbc:
        if use_library_rsa:
            encrypted_samples, init_vector = rsa_lib_wrapper.encrypt_cbc(samples_as_bytes,
                                                                         rsa.PublicKey(encryption_data.n, encryption_data.e))
        else:
            encrypted_samples, init_vector, block_leftover_len = rsa_wrapper.encrypt_cbc(samples_as_bytes,
                                                                                         encryption_data)
            encryption_data.block_leftover_len = block_leftover_len
        encryption_data.init_vector = init_vector
    else:
        if use_library_rsa:
            encrypted_samples = rsa_lib_wrapper.encrypt_ecb(samples_as_bytes,
                                                            rsa.PublicKey(encryption_data.n, encryption_data.e))
        else:
            encrypted_samples, block_leftover_len = rsa_wrapper.encrypt_ecb(samples_as_bytes, encryption_data)
            encryption_data.block_leftover_len = block_leftover_len

    encryption_utils.write_rsa_data_to_file(encryption_data_file_name, encryption_data)

file = open("nowy.wav", "wb")
riffChunk.write(file)
fmtChunk.write(file)
dataChunk.write(file, fmtChunk, encrypted_samples)
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
