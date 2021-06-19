import wav_chunks
import rsa
import encryption_utils


# def encrypt_message(message: bytes):
#     message_array = bytearray(message)
#
#     key_bits = 10
#     block_size = key_bits // 16 - 1
#
#     public_key, private_key = rsa.newkeys(key_bits)
#     number_of_blocks, block_leftover_len = divmod(len(message_array), block_size)
#
#     blocks = []
#     for i in range(0, number_of_blocks - 1):
#         block = message_array[i * block_size: i * block_size + block_size]
#         block = rsa.encrypt(block, public_key)
#         blocks.append(block)
#     if block_leftover_len > 0:
#         leftover = message_array[len(message_array) - 1 - block_leftover_len: len(message_array)]
#         leftover = rsa.encrypt(leftover, public_key)
#         blocks.append(leftover)
#     encrypted_message = bytes(*blocks)
#     return encrypted_message, private_key, block_size


def encrypt_ebc(message: bytes, public_key: rsa.PublicKey) -> (bytes, int):
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
    return encrypted_message, block_size


def decrypt_ebc(message: bytes, private_key: rsa.PrivateKey) -> bytes:
    message_array = bytearray(message)
    #number_of_blocks, block_leftover_len = divmod(len(message_array), block_size)
    number_of_blocks = int(len(message_array) / int(private_key.n.bit_length() / 8))
    block_size = int(private_key.n.bit_length() / 8)
    decrypted_blocks = []
    for i in range(0, number_of_blocks - 1):
        block = message_array[i * block_size: i * block_size + block_size]
        block = rsa.decrypt(block, private_key)
        decrypted_blocks.append(block)
    decrypted_message = b"".join(decrypted_blocks)
    return decrypted_message


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
    crypto, bs = encrypt_ebc(bytearray(message.encode("utf-8")), pub)
    decrypted = decrypt_ebc(crypto, priv)
    print(decrypted)
