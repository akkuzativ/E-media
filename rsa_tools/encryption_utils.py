import yaml
import secrets


class RsaData:
    def __init__(self, n: int, e: int, d: int, p: int, q: int, init_vector: int = None):
        self.e = e
        self.d = d
        self.n = n
        self.p = p
        self.q = q
        self.init_vector = init_vector

    def __iter__(self):
        return iter((self.n, self.e, self.d, self.p, self.q))


def read_rsa_data_from_file(file_name: str) -> RsaData:
    with open(file_name, "r") as file:
        loaded_dict = yaml.load(stream=file, Loader=yaml.SafeLoader)
    if loaded_dict is not None:
        n = loaded_dict["n"]
        e = loaded_dict["e"]
        d = loaded_dict["d"]
        p = loaded_dict["p"]
        q = loaded_dict["q"]
        init_vector = loaded_dict["init_vector"]
        return RsaData(n, e, d, p, q, init_vector)
    else:
        raise Exception


def write_rsa_data_to_file(file_name: str, data: RsaData) -> None:
    with open(file_name, "w+") as file:
        data_dict = {
            "n": data.n,
            "e": data.e,
            "d": data.d,
            "p": data.p,
            "q": data.q,
            "init_vector": data.init_vector
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
