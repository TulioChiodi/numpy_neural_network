from utils import *
import pytest
import pandas as pd
from pathlib import Path

@pytest.fixture(scope='module')
def df():
    df = pd.read_csv('instruments.csv')
    fname_list = df['fname'].to_list()
    path_in = Path('dataset')
    f_list = [path_in/i for i in fname_list]
    return f_list

def test_read_dataset(df):
        assert read_dataset() == df 


