import yaml


class RsaData:
    class PublicKey:
        def __init__(self, n: int, e: int):
            self.n = n
            self.e = e

    class PrivateKey:
        def __init__(self, n: int, d: int):
            self.n = n
            self.d = d

    def __init__(self, public_key: PublicKey, private_key: PrivateKey, block_size: int):
        self.private_key = private_key
        self.public_key = public_key
        self.block_size = block_size


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
        return RsaData(public_key, private_key, block_size)
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
