# measure_lzc_energy.py
import numpy as np
import subprocess

INPUT_FILE = "lzc_input.txt"
RAW_METRICS = "lzc_raw_metrics.txt"
ENERGY_TABLE = "lzc_energy_table.txt"

# QEMU = "qemu-aarch64"        # or qemu-arm
QEMU = None
# C_BINARY = "./lzc_qemu"      # compiled C harness
C_BINARY = "./lzc_macos"

# example MCU energy model
ENERGY_PER_CYCLE = 3.2e-12   # joules per cycle (example)


def load_dataset():
    """
    Returns data of shape [N, T, F] with 0/1 values.
    Replace this with real UCI HAR loading.
    """
    # dummy example
    np.random.seed(0)
    data = np.random.randint(0, 2, size=(10, 128, 1))
    return data


def write_lzc_input(data):
    with open(INPUT_FILE, "w") as f:
        for sample in data:
            flat = sample.reshape(-1)
            line = "".join(str(int(x)) for x in flat)
            f.write(line + "\n")

''''
def run_qemu():
    subprocess.run(
        [QEMU, C_BINARY, INPUT_FILE, RAW_METRICS],
        check=True
    )
'''

def run_qemu():
    cmd = [C_BINARY, INPUT_FILE, RAW_METRICS]
    subprocess.run(cmd, check=True)


def metrics_to_energy():
    with open(RAW_METRICS) as fin, open(ENERGY_TABLE, "w") as fout:
        for line in fin:
            cycles, instr = map(int, line.split())
            joules = cycles * ENERGY_PER_CYCLE
            fout.write(f"{joules}\n")


def main():
    data = load_dataset()
    write_lzc_input(data)
    run_qemu()
    metrics_to_energy()


if __name__ == "__main__":
    main()