from utils import RSA
from utils.encryption_utils import *


def new_keys(bit_length: int):
    public, private, primes = RSA.choose_prime_numbers(bit_length // 2)
    e = public[0]
    n = public[1]
    d = private[0]
    p = primes[0]
    q = primes[1]
    return RsaData(n, e, d, p, q)


def encrypt_block(message: bytes, rsa_data: RsaData) -> bytes:
    message_as_number = int.from_bytes(message, byteorder="little")
    encrypted_message_as_number = pow(message_as_number, rsa_data.e, rsa_data.n)
    a = encrypted_message_as_number.bit_length()
    b = rsa_data.n.bit_length()
    encrypted_message = int.to_bytes(encrypted_message_as_number, length=(rsa_data.n.bit_length()//8+1),
                                     byteorder="little")
    return encrypted_message


def decrypt_block(message: bytes, rsa_data: RsaData):
    message_as_number = int.from_bytes(message, byteorder="little")
    decrypted_message_as_number = pow(message_as_number, rsa_data.d, rsa_data.n)
    decrypted_message = int.to_bytes(decrypted_message_as_number, length=(rsa_data.n.bit_length()//8),
                                     byteorder="little")
    return decrypted_message


def encrypt_ecb(message: bytes, rsa_data: RsaData) -> (bytes, int):
    message_array = bytearray(message)
    block_size = (rsa_data.n.bit_length()//8) - 1
    blocks = divide_data_into_blocks(message_array, block_size)
    encrypted_blocks = list()
    for block in blocks:
        encrypted_blocks.append(encrypt_block(block, rsa_data))
    encrypted_message = b"".join(encrypted_blocks)
    block_leftover_len = len(blocks[len(blocks)-1])
    if block_leftover_len == block_size:
        block_leftover_len = 0
    return encrypted_message, block_leftover_len


def decrypt_ecb(message: bytes, rsa_data: RsaData, block_leftover_len: int) -> bytes:
    message_array = bytearray(message)
    original_block_size = (rsa_data.n.bit_length()//8) - 1
    block_size = int(rsa_data.n.bit_length() / 8) + 1
    blocks = divide_data_into_blocks(message_array, block_size)
    decrypted_blocks = list()
    for block in blocks:
        decrypted_blocks.append(decrypt_block(block, rsa_data)[0:original_block_size])
    if block_leftover_len > 0:
        decrypted_blocks[len(decrypted_blocks)-1] = decrypted_blocks[len(decrypted_blocks)-1][0:block_leftover_len]
    decrypted_message = b"".join(decrypted_blocks)
    return decrypted_message


def encrypt_cbc(message: bytes, rsa_data: RsaData) -> (bytes, int, int):
    message_array = bytearray(message)
    block_size = (rsa_data.n.bit_length()//8) - 1
    init_vector = create_random_init_vector(rsa_data.n.bit_length())
    previous_vector = init_vector.to_bytes(length=(rsa_data.n.bit_length()//8+1), byteorder="little")
    blocks = divide_data_into_blocks(message_array, block_size)
    encrypted_blocks = list()
    for block in blocks:
        block_as_number = int.from_bytes(block, "little")
        previous_vector_as_number = int.from_bytes(previous_vector[0:len(block)], byteorder="little")
        block = (block_as_number ^ previous_vector_as_number).to_bytes(length=block_size, byteorder="little")
        encrypted_block = encrypt_block(block, rsa_data)
        encrypted_blocks.append(encrypted_block)
        previous_vector = encrypted_block
    encrypted_message = b"".join(encrypted_blocks)
    block_leftover_len = len(blocks[len(blocks)-1])
    if block_leftover_len == block_size:
        block_leftover_len = 0
    return encrypted_message, init_vector, block_leftover_len


def decrypt_cbc(message: bytes, rsa_data: RsaData, init_vector: int, block_leftover_len: int) -> bytes:
    message_array = bytearray(message)
    original_block_size = (rsa_data.n.bit_length() // 8) - 1
    block_size = int(rsa_data.n.bit_length() / 8) + 1
    previous_vector = init_vector.to_bytes(length=(rsa_data.n.bit_length()//8+1), byteorder="little")
    blocks = divide_data_into_blocks(message_array, block_size)
    decrypted_blocks = list()
    for block in blocks:
        decrypted_block = decrypt_block(block, rsa_data)
        previous_vector_as_number = int.from_bytes(previous_vector[0:len(decrypted_block)], byteorder="little")
        decrypted_block_as_number = int.from_bytes(decrypted_block, "little")
        decrypted_block = (decrypted_block_as_number ^ previous_vector_as_number).to_bytes(length=len(decrypted_block),
                                                                                           byteorder="little")
        decrypted_blocks.append(decrypted_block[0:original_block_size])
        previous_vector = block
    if block_leftover_len > 0:
        decrypted_blocks[len(decrypted_blocks)-1] = decrypted_blocks[len(decrypted_blocks)-1][0:block_leftover_len]
    decrypted_message = b"".join(decrypted_blocks)
    return decrypted_message