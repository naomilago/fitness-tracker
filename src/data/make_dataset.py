from loguru import logger
from glob import glob
import pandas as pd

from typing import (
  Tuple, 
  Union,
  List, 
  Set, 
)

# Defining the reading process

def data_reader(files:  Union[List[str], Tuple[str], Set[str]]) -> Tuple[pd.core.frame.DataFrame]:
  
  '''
  Reads the data files within a path and returns two DataFrames referring to the type of measurement made.
  It also extract and handles relevant features from the file names. 
    
  Parameters:
  - files (Union[List[str], Tuple[str], Set[str]]):
    A collection of paths to data files in CSV format.
    
  Returns:
  - Tuple[pd.core.frame.DataFrame]:
    A tuple containing two Pandas DataFrames:
        - The accelerometer data in the first DataFrame.
        - The gyroscope data in the second DataFrame.
        
  Notes:
  - Accelerometer x Gyroscope
    These are the two frequency of measurements.
    
    Acceleriometer -> 15.500Hz
    Gyroscope -> 25.000Hz
  '''
  
  acc_df = pd.DataFrame()
  gyr_df = pd.DataFrame()

  acc_set, gyr_set = 1, 1

  for _ in files:
    df = pd.read_csv(_)
    
    _ = str.replace(_, '/workspaces/fitness-tracker', '../..')
    category = str.rstrip(str.rstrip(str.split(str.split(_, '-')[2], '_')[0], '123'), '_MetaWear_2019')
    participant = str.replace(str.split(_, '-')[0], raw_data_path('rel')[:-5], '')
    label = str.split(_, '-')[1]
    
    df['participant'], df['label'], df['category'] = participant, label, category
    
    if 'Accelerometer' in _:
      df['set'] = acc_set
      acc_set += 1
      acc_df = pd.concat([acc_df, df], axis=0)
    else:
      df['set'] = gyr_set
      gyr_set += 1
      gyr_df = pd.concat([gyr_df, df], axis=0)
      
  acc_df.index = pd.to_datetime(acc_df['epoch (ms)'], unit='ms')
  gyr_df.index = pd.to_datetime(gyr_df['epoch (ms)'], unit='ms')

  del acc_df['epoch (ms)']
  del acc_df['time (01:00)']
  del acc_df['elapsed (s)']

  del gyr_df['epoch (ms)']
  del gyr_df['time (01:00)']
  del gyr_df['elapsed (s)']
  
  return acc_df, gyr_df

# Getting the raw data

raw_data_path = lambda x: '/workspaces/fitness-tracker/data/raw/MetaMotion/*.csv' if x == 'abs' else '../../data/raw/MetaMotion/*.csv'
files = glob(raw_data_path('abs'))

# Reading, structuring, and merging the raw data

logger.info('Reading the raw data...')

acc_df, gyr_df = data_reader(files)

logger.info('Merging the two different methods...')

merged_data = pd.concat([acc_df.iloc[:, :3], gyr_df], axis=1)
merged_data.columns = [
  'acc_x', 'acc_y', 'acc_z',
  'gyr_x', 'gyr_y', 'gyr_z',
  'participant', 'label', 'category', 'set'
]

# Resampling the data by handling the granularity

logger.info('Resampling the data and handling gaps...')

sampling = {
  'acc_x': 'mean', 
  'acc_y': 'mean', 
  'acc_z': 'mean',
  'gyr_x': 'mean', 
  'gyr_y': 'mean', 
  'gyr_z': 'mean',
  'label': 'last', 
  'category': 'last', 
  'participant': 'last', 
  'set': 'last',
}

merged_data[:1000].resample(rule='200ms').apply(sampling)

days = [g for _, g in merged_data.groupby(pd.Grouper(freq='D'))]

resample = pd.concat([df.resample(rule='200ms').apply(sampling).dropna() for df in days])
resample['set'] = resample['set'].astype('int')

# Exporting the main dataset

resample.to_pickle('/workspaces/fitness-tracker/data/interim/01_processed_data.pkl')

logger.success('The dataset was made and successfully exported 🎉')