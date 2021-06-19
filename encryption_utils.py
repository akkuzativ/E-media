import yaml
import secrets


class RsaData:
    class PublicKey:
        def __init__(self, n: int, e: int):
            self.n = n
            self.e = e

    class PrivateKey:
        def __init__(self, n: int, d: int):
            self.n = n
            self.d = d

    def __init__(self, public_key: PublicKey, private_key: PrivateKey, block_size: int, init_vector: int = None):
        self.private_key = private_key
        self.public_key = public_key
        self.block_size = block_size
        self.init_vector = init_vector


def read_rsa_data_from_file(file_name: str) -> RsaData:
    with open(file_name, "r") as file:
        loaded_dict = yaml.load(stream=file, Loader=yaml.SafeLoader)
    if loaded_dict is not None:
        public_key = RsaData.PublicKey(
            loaded_dict["n"],
            loaded_dict["e"]
        )
        private_key = RsaData.PrivateKey(
            loaded_dict["n"],
            loaded_dict["d"]
        )
        block_size = loaded_dict["block_size"]
        init_vector = loaded_dict["init_vector"]
        return RsaData(public_key, private_key, block_size, init_vector)
    else:
        raise Exception


def write_rsa_data_to_file(file_name: str, data: RsaData) -> None:
    with open(file_name, "w+") as file:
        data_dict = {
            "block_size": data.block_size,
            "n": data.public_key.n,
            "e": data.public_key.e,
            "d": data.private_key.d,
        }
        yaml.dump(data=data_dict, Dumper=yaml.Dumper, stream=file, sort_keys=False)


def divide_data_into_blocks(message: bytearray, preferred_block_size: int) -> list:
    message_array = bytearray(message)
    number_of_blocks, block_leftover_len = divmod(len(message_array), preferred_block_size)
    blocks = []
    for i in range(0, number_of_blocks):
        block = message_array[i * preferred_block_size: i * preferred_block_size + preferred_block_size]
        blocks.append(block)
    if block_leftover_len > 0:
        leftover = message_array[len(message_array) - block_leftover_len: len(message_array) + 1]
        blocks.append(leftover)
    return blocks


def create_random_init_vector(bit_length: int) -> int:
    return secrets.randbits(bit_length)
