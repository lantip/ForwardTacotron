from pathlib import Path

from textgrids import TextGrid
import argparse
import numpy as np


class Processor:

    def __init__(self, tar_dir):
        self.tar_dir = tar_dir

    def save_durations(self, path):
        tg_reader = TextGrid()
        tg_reader.read(path)
        di = tg_reader['phones']
        phons = [d.text for d in di]
        durs = [d.dur for d in di]
        res = list(zip(phons, durs))
        np.save((self.tar_dir / path.stem).with_suffix('.npy'), res)


if __name__ == '__main__':
    input_dir = '/Users/cschaefe/Downloads/montreal-forced-aligner/mfa_outputs/asvoice2_mfa_flat'
    output_dir = '/Users/cschaefe/Downloads/montreal-forced-aligner/mfa_outputs/mfa_durations'
    processor = Processor(tar_dir=Path(output_dir)).save_durations
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    file_list = [file for file in Path(input_dir).iterdir() if file.suffix == '.TextGrid']
    for f in file_list:
        processor(f)
    print(file_list)