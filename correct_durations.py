import argparse
import numpy as np
from utils import hparams as hp
from utils.files import unpickle_binary
from utils.paths import Paths

if __name__ == '__main__':
    # Parse Arguments
    parser = argparse.ArgumentParser(description='Train Tacotron TTS')
    parser.add_argument('--force_train', '-f', action='store_true', help='Forces the model to train past total steps')
    parser.add_argument('--force_gta', '-g', action='store_true', help='Force the model to create GTA features')
    parser.add_argument('--force_align', '-a', action='store_true', help='Force the model to create attention alignment features')
    parser.add_argument('--force_cpu', '-c', action='store_true', help='Forces CPU-only training, even when in CUDA capable environment')
    parser.add_argument('--extract_pitch', '-p', action='store_true', help='Extracts phoneme-pitch values only')
    parser.add_argument('--hp_file', metavar='FILE', default='hparams.py', help='The file to use for the hyperparameters')
    args = parser.parse_args()

    hp.configure(args.hp_file)  # Load hparams from file
    paths = Paths(hp.data_path, hp.voc_model_id, hp.tts_model_id)

    train_data = unpickle_binary(paths.data / 'train_dataset.pkl')
    val_data = unpickle_binary(paths.data / 'val_dataset.pkl')
    dataset = train_data + val_data
    text_dict = unpickle_binary(paths.data / 'text_dict.pkl')

    hop_factor = hp.sample_rate / hp.hop_length

    for item_id, mel_len in dataset:
        text = text_dict[item_id]
        dur = np.load(paths.alg / f'{item_id}.npy', allow_pickle=False)
        voice_mask = np.load(paths.voice_mask / f'{item_id}.npy', allow_pickle=False)
        dur_cum = np.cumsum(np.pad(dur, (1, 0)))



        for i in range(dur):
            voice_interval = voice_mask[dur_cum[i]:dur_cum[i+1]]
            print()

