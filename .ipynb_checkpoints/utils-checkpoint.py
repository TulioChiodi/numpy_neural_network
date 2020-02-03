import pandas as pd
from pydub import AudioSegment
from pathlib import Path
from tqdm import tqdm
import os


def read_dataset(read_path):
    df = pd.read_csv('instruments.csv')
    fname_list = df['fname'].to_list()
    f_list = [read_path/i for i in fname_list]
    return f_list

def split_audio(file_list, read_path, write_path='splitted_dataset', chunk_length=2000):
    write_path = Path(read_path/write_path)
    for file in tdqm(file_list):
        audio = AudioSegment.from_wav(file)
        if audio.duration_seconds > (chunk_length/1000):
            os.remove(file)
            chunks = make_chunks(audio, chunk_length)
            for i, chunk in enumerate(chunks):
                chunk_name = ("{os.path.splitext(os.path.basename(file))[0]}_chunk{i+1}of{len(chunks)}.wav")
                chunk.export(write_path/chunk_name, format='wav')  
