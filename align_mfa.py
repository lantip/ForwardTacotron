from pathlib import Path

from textgrids import TextGrid
import argparse
import numpy as np

def get_text(file):
    with open(file, 'r', encoding='utf-8') as f:
        return f.readlines()[0]


if __name__ == '__main__':
    text_dir = '/Users/cschaefe/datasets/audio_data/asvoice2_mfa_flat'
    dur_dir = '/Users/cschaefe/Downloads/montreal-forced-aligner/mfa_outputs/mfa_durations'

    dur_files = Path(dur_dir).glob('**/*.npy')

    for dur_file in dur_files:
        dur_list = np.load(dur_file)
        dur_text = dur_list[:, 0]
        for i in range(len(dur_text)):
            dur_text[i] = dur_text[i].replace('\n', '').replace('sil', ' ').replace('sp', ' ')
        dur_cum = np.cumsum(dur_list[:, 1].astype(np.float), axis=0) * 22050. / 256.
        text = get_text(Path(text_dir) / f'{dur_file.stem}.lab')
        char_indices = dict()
        durations = []

        if dur_text[0] == ' ':
            j_start = 1
        else:
            j_start = 0
        j = j_start
        try:
            for i, c in enumerate(text):
                if dur_text[j] == c:
                    if j > j_start:
                        dur = dur_cum[j] - dur_cum[j-1]
                    else:
                        dur = dur_cum[j]
                    j = j+1
                else:
                    dur = 0.
                durations.append(dur)
        except Exception as e:
            print(e)
        durations[-1] = dur_cum[-1] - sum(durations)
        print(durations)
        print(f'sum dur {sum(durations)} dur cum {dur_cum[-1]} ')
