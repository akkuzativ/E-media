import audioop
import struct
import math

import wav_chunks
import rsa


class EncryptionInfo:
    def __init__(self, private_key, public_key, block_size):
        self.private_key = private_key
        self.public_key = public_key
        self.block_size = block_size


def encrypt_message(message: bytes):
    message_array = bytearray(message)

    key_bits = 512
    block_size = key_bits // 16 - 1

    public_key, private_key = rsa.newkeys(key_bits)
    number_of_blocks, block_leftover_len = divmod(len(message_array), block_size)

    blocks = []
    for i in range(0, number_of_blocks - 1):
        block = message_array[i * block_size: i * block_size + block_size]
        block = rsa.encrypt(block, public_key)
        blocks.append(block)
    if block_leftover_len > 0:
        leftover = message_array[len(message_array) - 1 - block_leftover_len: len(message_array)]
        leftover = rsa.encrypt(leftover, public_key)
        blocks.append(leftover)
    encrypted_message = bytes(*blocks)
    return encrypted_message, private_key, block_size


def decrypt_message(message: bytes, block_size, public_key) -> bytes:
    pass


def encrypt_data_chunk(data_chunk: wav_chunks.DataChunk, fmt_chunk: wav_chunks.FmtChunk) -> wav_chunks.DataChunk:
    message = wav_chunks.DataChunk.Contents.channels_to_bytes(fmt_chunk, data_chunk.data)
    pass


def decrypt_data_chunk(data_chunk: wav_chunks.DataChunk, fmt_chunk: wav_chunks.FmtChunk, block_size,
                       public_key) -> wav_chunks.DataChunk:
    message = wav_chunks.DataChunk.Contents.channels_to_bytes(fmt_chunk, data_chunk.data)
    pass


if __name__ == "__main__":
    encrypt_message(int.to_bytes(112341235, byteorder="little", signed=True, length=4))
