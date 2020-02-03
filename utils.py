from shutil import copy2
import librosa
from glob import glob
import soundfile as sf
import numpy as np
import pandas as pd
from pydub import AudioSegment
from pydub.utils import make_chunks
from pathlib import Path
from tqdm import tqdm
import os


def read_dataset(csv_path='instruments.csv',dataset_path=Path('dataset')):
    df = pd.read_csv(csv_path)
    fname_list = df['fname'].to_list()
    f_list = [dataset_path/i for i in fname_list]
    return f_list

def split_audio(file_list, splitted_dataset_path='splitted_dataset', chunk_length=2000):
    splitted_dataset_path = Path(splitted_dataset_path)
    splitted_dataset_path.mkdir(exist_ok=True)
    for file in tqdm(file_list):
        audio = AudioSegment.from_wav(file)
        if audio.duration_seconds > (chunk_length/1000):
            chunks = make_chunks(audio, chunk_length)
            for i, chunk in enumerate(chunks):
                chunk_name = (f"{os.path.splitext(os.path.basename(file))[0]}_chunk{i+1}of{len(chunks)}.wav")
                chunk.export(splitted_dataset_path/chunk_name, format='wav')  
        else:
            copy2(file,splitted_dataset_path)

def get_wav_list(dataset_path=Path('splitted_dataset')):
    return glob(str(dataset_path/'*.wav'))    


def zero_padding(file_list, chunk_length=2000):
    n_samples = int((chunk_length/1000)*44100)
    for file in tqdm(file_list):
        signal, sr = librosa.load(file, sr=None)
        if len(signal) < n_samples:
            zero_arr_length = n_samples - len(signal) 
            signal_padded = np.pad(signal,(0,zero_arr_length))
            sf.write(file, signal_padded, sr)

def resampling(file_list, sr=8000):
    for file in tqdm(file_list):
        signal, sr = librosa.load(file, sr=sr)
        sf.write(file, signal, sr)


