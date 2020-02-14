from shutil import copy2
import h5py
import librosa
from glob import glob
import soundfile as sf
import numpy as np
import pandas as pd
from pydub import AudioSegment
from pydub.utils import make_chunks
from pathlib import Path
from tqdm import tqdm
import wave
import os


def read_dataset(csv_path='instruments.csv',dataset_path='dataset'):
    dataset_path = Path(dataset_path)
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

def get_wav_list(dataset_path='splitted_dataset'):
    dataset_path = Path(dataset_path)
    return glob(str(dataset_path/'*.wav'))    


def zero_padding(file_list, chunk_length=2000):
    n_samples = int((chunk_length/1000)*44100)
    for file in tqdm(file_list):
        signal, sr = librosa.load(file, sr=None)
        if len(signal) < n_samples:
            zero_arr_length = n_samples - len(signal) 
            signal_padded = np.pad(signal,(0,zero_arr_length))
            sf.write(file, signal_padded, sr)

def resampling(file_list, sr=16000):
    for file in tqdm(file_list):
        signal, sr = librosa.load(file, sr=sr)
        sf.write(file, signal, sr)


def test_dataset_setup():
    splittedfile_list = get_wav_list(dataset_path='splitted_dataset')
    file_list = read_dataset(csv_path='instruments.csv', dataset_path='dataset')
    file_length_seconds = []
    framerate_list = []
    file_length_seconds2 = []
    framerate_list2 = []
    for file in splittedfile_list:
        with wave.open(f'/home/tuliochiodi/workspace/nn_numpy/project/{file}', 'rb') as wave_file:
            framerate =  wave_file.getframerate()
            nframes = wave_file.getnframes()
            framerate_list.append(framerate)
            file_length_seconds.append(nframes/framerate)

    for file2 in file_list:
        with wave.open(f'/home/tuliochiodi/workspace/nn_numpy/project/{file2}', 'rb') as wave_file:
            framerate =  wave_file.getframerate()
            nframes = wave_file.getnframes()
            framerate_list2.append(framerate)
            file_length_seconds2.append(nframes/framerate)

    assert framerate_list.count(framerate_list[0]) == len(framerate_list), f'Existem arquivos com framerate diferentes de {framerate_list[0]}'
    assert file_length_seconds.count(file_length_seconds[0]) == len(file_length_seconds), f'Existem arquivos com tamanho diferente de {file_length_seconds[0]}'
    print(f'------\nSplitted dataset:\n'
        f'framerate: {framerate_list[0]}Hz (all files has the same framerate)\n'
        f'File length (seconds): {file_length_seconds[0]}s (all files has the same length)\n'
        f'Total dataset length (seconds): {sum(file_length_seconds)}s\n')


    assert framerate_list2.count(framerate_list2[0]) == len(framerate_list2), f'Existem arquivos com framerate diferentes de {framerate_list2[0]}'
    print(f'Original dataset:\n'
        f'framerate: {framerate_list2[0]}Hz (all files has the same length)\n------')

def get_filename_list(splitted_dataset_path):
    path = Path(splitted_dataset_path)
    path_list = glob(str(path/'*.wav'))
    filename_list = [''.join([os.path.basename(file).split('_')[0],'.wav']) if len(file)>38 else os.path.basename(file) for file in path_list]
    np_filename_list = np.string_(filename_list)
    return filename_list, np_filename_list

def get_label_list(csv_path, filename_list):
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)
    label_list_orig = list(df['label'])
    fname_list_orig = list(df['fname'])
    label_list = [label_list_orig[fname_list_orig.index(file)] for file in filename_list] 
    bool_label_list = [ label=='Clarinet' for label in label_list ]
    return bool_label_list, label_list

def get_one_hot():
    return np.string_(['Non-clarinet', 'Clarinet'])

def get_signal_list(splitted_file_list):
    signal_list=[]
    for file in splitted_file_list:
        x,_ = librosa.load(file, sr=None)
        signal_list.append(librosa.util.normalize(x))
    return signal_list

def normalize(X, min=0.0, max=1.0):
    X_std = (X - X.min()) / (X.max() - X.min())
    X_scaled = X_std * (max - min) + min
    return X_scaled

def transf_melspec(signal_list, sr=16000, n_fft=1024, hop_length=512, n_mels=128, fmax=8000):
    spec_flatten_list = []
    spec_list = []
    for file in signal_list:
        S = librosa.feature.melspectrogram(file,
                    sr=sr,
                    n_fft=n_fft,
                    hop_length=hop_length,
                    n_mels=n_mels,
                    fmax=fmax
                    )
        S_log = np.log(S + 1e-9)
        S_log_scaled = normalize(S_log)
        S_log_scaled_flatten = S_log_scaled.flatten()
        spec_flatten_list.append(S_log_scaled_flatten)
        spec_list.append(S_log_scaled)
    return spec_flatten_list, spec_list

def save_in_hdf5(np_filename_list, bool_label_list, one_hot_labels, spec_flatten_list, etc):
    with h5py.File("dataset.hdf5", "w") as hdf:
        G = hdf.create_group('Dataset')
        G.create_dataset('file_names', data = np_filename_list)
        G.create_dataset('bool_labels', data = bool_label_list)
        G.create_dataset('one_hot_labels', data = one_hot_labels)
        G.create_dataset('spec_flatten_list', data = spec_flatten_list)
        G.create_dataset('etc', data = np.array(etc, dtype='i,2i'))


def setup_dataset(csv_path='instruments.csv', dataset_path='dataset', splitted_dataset_path='splitted_dataset', chunk_length=2, sr=16000):
    chunk_length = chunk_length*1000
    file_list = read_dataset(csv_path=csv_path, dataset_path=dataset_path)
    split_audio(file_list=file_list, splitted_dataset_path=splitted_dataset_path, chunk_length=chunk_length)
    splitted_file_list = get_wav_list(dataset_path=splitted_dataset_path)
    zero_padding(file_list=splitted_file_list, chunk_length=chunk_length)
    resampling(file_list=splitted_file_list, sr=sr)
    test_dataset_setup()
    print('Your dataset is ready! :D')
    filename_list, np_filename_list =  get_filename_list(splitted_dataset_path) 
    bool_label_list, _ = get_label_list(csv_path, filename_list)
    one_hot_labels = get_one_hot()
    signal_list = get_signal_list(splitted_file_list)
    spec_flatten_list, spec_list = transf_melspec(signal_list, sr=sr, fmax=sr/2)
    spec_shape = spec_list[0].shape
    save_in_hdf5(np_filename_list, bool_label_list, one_hot_labels, spec_flatten_list, (sr, spec_shape))
    return np_filename_list, bool_label_list, one_hot_labels, spec_flatten_list, (sr, spec_shape), signal_list, splitted_file_list

def read_hdf5():
    with h5py.File("dataset.hdf5","r") as hdf:
        name_lst_enc = np.array(hdf['/Dataset/file_names'])
        label_lst = np.array(hdf['Dataset/bool_labels'])
        one_hot_lst_enc = np.array(hdf['Dataset/one_hot_labels'])
        spec_flatten_lst = np.array(hdf['Dataset/spec_flatten_list'])
        etc = np.array(hdf['Dataset/etc'])

    name_lst = [file.decode("utf-8") for file in name_lst_enc]
    one_hot_lst = [file.decode("utf-8") for file in one_hot_lst_enc]
    sample_rate, shape = etc.tolist()
    wav_lst = get_wav_list()
    signal_list = get_signal_list(wav_lst)
    return  name_lst, label_lst, one_hot_lst, spec_flatten_lst, sample_rate, shape, signal_list


