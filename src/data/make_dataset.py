import sys
sys.path.append('/workspaces/fitness-tracker/')

from config.utils import *

# Getting the raw data

files = glob(get_raw_data_path('abs'))

# Reading, structuring, and merging the raw data

logger.info('Reading the raw data...')

acc_df, gyr_df = data_reader(files, verbose=False)

logger.info('Merging the two different methods...')

merged_data = pd.concat([acc_df.iloc[:, :3], gyr_df], axis=1)
merged_data.columns = pd.Index([
  'acc_x', 'acc_y', 'acc_z',
  'gyr_x', 'gyr_y', 'gyr_z',
  'participant', 'label', 'category', 'set'
])

# Resampling the data by handling the granularity

logger.info('Resampling the data and handling gaps...')

sampling: Any = {
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

days = [g for _, g in merged_data.groupby(pd.Grouper(freq='D'))]

resample = pd.concat([pd.DataFrame(df.resample(rule='200ms').apply(sampling).dropna()) for df in days])
resample['set'] = resample['set'].astype('int')

# Exporting the main dataset

resample.to_pickle('/workspaces/fitness-tracker/data/interim/01_processed_data.pkl')

logger.success('The dataset was made and successfully exported 🎉')