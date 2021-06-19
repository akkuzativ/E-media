import wav_chunks
import rsa


from encryption_utils import *


def encrypt_ebc(message: bytes, public_key: rsa.PublicKey) -> bytes:
    message_array = bytearray(message)
    block_size = int(public_key.n.bit_length() / 8) - 11
    number_of_blocks, block_leftover_len = divmod(len(message_array), block_size)
    encrypted_blocks = []
    for i in range(0, number_of_blocks - 1):
        block = message_array[i * block_size: i * block_size + block_size]
        block = rsa.encrypt(block, public_key)
        encrypted_blocks.append(block)
    if block_leftover_len > 0:
        leftover = message_array[len(message_array) - 1 - block_leftover_len: len(message_array) + 1]
        leftover = rsa.encrypt(leftover, public_key)
        encrypted_blocks.append(leftover)
    encrypted_message = b"".join(encrypted_blocks)
    return encrypted_message


def decrypt_ebc(message: bytes, private_key: rsa.PrivateKey) -> bytes:
    message_array = bytearray(message)
    number_of_blocks = int(len(message_array) / int(private_key.n.bit_length() / 8))
    block_size = int(private_key.n.bit_length() / 8)
    decrypted_blocks = []
    for i in range(0, number_of_blocks):
        block = message_array[i * block_size: i * block_size + block_size]
        block = rsa.decrypt(block, private_key)
        decrypted_blocks.append(block)
    decrypted_message = b"".join(decrypted_blocks)
    return decrypted_message


def encrypt_cbc(message: bytes, public_key: rsa.PublicKey) -> (bytes, int):
    message_array = bytearray(message)
    block_size = int(public_key.n.bit_length() / 8) - 11
    init_vector = create_random_init_vector(public_key.n.bit_length())
    previous_vector = init_vector.to_bytes(length=(public_key.n.bit_length()//8), byteorder="little")
    blocks = divide_data_into_blocks(message_array, block_size)
    encrypted_blocks = list()
    for block in blocks:
        block_as_number = int.from_bytes(block, "little")
        previous_vector_as_number = int.from_bytes(previous_vector[0:len(block)], byteorder="little")
        block = (block_as_number ^ previous_vector_as_number).to_bytes(length=len(block), byteorder="little")
        encrypted_block = rsa.encrypt(block, public_key)
        encrypted_blocks.append(encrypted_block)
        previous_vector = encrypted_block
    encrypted_message = b"".join(encrypted_blocks)
    return encrypted_message, init_vector


def decrypt_cbc(message: bytes, private_key: rsa.PrivateKey, init_vector: int) -> bytes:
    message_array = bytearray(message)
    block_size = int(private_key.n.bit_length() / 8)
    previous_vector = init_vector.to_bytes(length=(private_key.n.bit_length()//8), byteorder="little")
    blocks = divide_data_into_blocks(message_array, block_size)
    decrypted_blocks = list()
    for block in blocks:
        decrypted_block = rsa.decrypt(block, private_key)
        previous_vector_as_number = int.from_bytes(previous_vector[0:len(block)], byteorder="little")
        decrypted_block_as_number = int.from_bytes(decrypted_block, "little")
        decrypted_block = (decrypted_block_as_number ^ previous_vector_as_number).to_bytes(length=len(block), byteorder="little")
        decrypted_blocks.append(decrypted_block)
        previous_vector = block
    encrypted_message = b"".join(decrypted_blocks)
    return encrypted_message


def encrypt_data_chunk(data_chunk: wav_chunks.DataChunk, fmt_chunk: wav_chunks.FmtChunk) -> wav_chunks.DataChunk:
    message = wav_chunks.DataChunk.Contents.channels_to_bytes(fmt_chunk, data_chunk.data)
    pass


def decrypt_data_chunk(data_chunk: wav_chunks.DataChunk, fmt_chunk: wav_chunks.FmtChunk, block_size,
                       public_key) -> wav_chunks.DataChunk:
    message = wav_chunks.DataChunk.Contents.channels_to_bytes(fmt_chunk, data_chunk.data)
    pass


if __name__ == "__main__":
    message = "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Nam tincidunt odio quis aliquet placerat. Curabitur varius odio blandit sollicitudin congue. Sed nec ex id leo lobortis facilisis nec vitae nunc. Etiam vulputate vitae enim quis pulvinar. Duis quis sollicitudin leo. Nunc et venenatis risus. Vivamus dui velit, egestas at lectus non, fringilla vehicula ligula. Aliquam a nisl sapien. Phasellus blandit nisi in nisi egestas accumsan. Morbi id rhoncus libero. Integer hendrerit diam est, at dapibus odio cursus ac. "
    pub, priv = rsa.newkeys(128)
    # crypto = encrypt_ebc(bytearray(message.encode("utf-8")), pub)
    # decrypted = decrypt_ebc(crypto, priv)
    # print(decrypted.decode("utf-8"))
    crypto, init = encrypt_cbc(bytearray(message.encode("utf-8")), pub)
    decrypted = decrypt_cbc(bytearray(message.encode("utf-8")), priv, init)
    print(decrypted.decode("utf-8"))
