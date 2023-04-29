import numpy as np
import struct

def create_npz(filename: str):
    ar_0 = np.arange(10, dtype=np.float32).reshape(2, 5)
    ar_1 = np.arange(20, dtype=np.float32).reshape(1, 2, 2, 5)
    arrays = {}
    arrays["in_0"] = ar_0
    arrays["in_1"] = ar_1

    np.savez(filename, **arrays)

def load_npz(filename: str):
    npz_file = np.load(filename)
    print(npz_file.files)
    for name in npz_file.files:
        print(npz_file[name])
    return npz_file

def write_bin(npz_file, bin_file):
    with open(bin_file, "wb") as f:
        for name in npz_file.files:
            array = npz_file[name].flatten()
            for n in array:
                f.write(struct.pack("<f", n))



if __name__ == "__main__":
    create_npz("test.npz")
    npz_file = load_npz("test.npz")
    write_bin(npz_file, "arrays.param")

