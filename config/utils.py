import matplotlib.pyplot as plt
from loguru import logger
import matplotlib as mpl
import seaborn as sns
from glob import glob
from typing import *
import pandas as pd
import requests
import neptune
import json
import os

# Visualization settings

plt.style.use('dark_background')
plt.rcParams['grid.color'] = '#212121'
plt.rcParams['figure.max_open_warning'] = 30

sns.set_palette("rocket_r")
color_pal = sns.color_palette()

# Function to fetch API keys from a secret manager (Infisical)
# - Used for experiment tracking on Neptune

def secret_fetcher(query: str, env: str = 'dev', secrets_path: str = '/workspaces/fitness-tracker/.secrets.json', verbose: bool = True) -> Union[Tuple[str, str], None]:
  
  '''
    Fetches a secret value from the Infisical (an encrypted platform for managing secrets) based on the provided query.
    
    Parameters:
    - query (str, required): 
      A string referring to the identifier for the secret to retrieve.
    - env (str, required):
      A string referring to the environment to fetch the secret from (default is `dev`).
    - secrets_path (str, opional):
      A string referring to the file path to a JSON file containing API secrets (default is `/workspaces/fitness-tracker/.secrets.json`).
    - verbose (bool, optional):
      A boolean variable to indicate whether showing logs or not (default is `True`).
        Example: When setting to production, it's strongly recommended in order to keep track of everything.
  
    Returns:
    - Union[Tuple[str, str], None]: A tuple containing the secret key and value if successful, or None in case of an error.
    
    Note:
      The function performs a POST request to acquire the API access key and then a GET request to fetch the desired secret.
  '''
  
  if query == '':
    logger.error('Please, insert a valid query!')
    return None
  
  with open(secrets_path) as f:
    secrets = json.load(f)

  payload = f"clientSecret={str(secrets['infisical']['clientSecret'])}&clientId={str(secrets['infisical']['clientId'])}"

  headers = {
    'Content-Type': 'application/x-www-form-urlencoded'
  }

  response = requests.request('POST', 'https://app.infisical.com/api/v1/auth/universal-auth/login', headers=headers, data=payload)

  if response.status_code == 200:
    if verbose:
      logger.success('API access keys fetching complete...')
      
    accessToken = response.json()['accessToken']
    
    url = f"https://app.infisical.com/api/v3/secrets/raw/{query}?environment={env}&workspaceId={str(secrets['infisical']['workspaceId'])}"
    
    payload = str()
    headers = {
      'Authorization': f'Bearer {accessToken}'
    }
    
    response = requests.request('GET', url, headers=headers, data=payload)
    
    if response.status_code == 200: 
      if verbose:
        logger.success('Secrets fetching complete...')
      
      key = response.json()['secret']['secretKey']
      value = response.json()['secret']['secretValue']
      
      return key, value
      
    else:
      if verbose:
        logger.error('Secrets fetching failed! Check out the details below:')
        
      raise Exception(f"{response.json()['statusCode']} - {response.json()['message']}")
  else:
    if verbose:
      logger.error('API access keys fetching failed! Check out the details below:')
      
    raise Exception(f"{response.json()['statusCode']} - {response.json()['message']}")
  
# Function to get the raw data path 
# - This is a support for the next one - that'll be handling the data files
  
def get_raw_data_path(sensor_type: str) -> str:
  
    '''
    Get the data path based on the preference (whether absolute or relative).

    Parameters:
    - sensor_type (str): 
      A string referring to the type of sensor.
        Options: 'abs' for absolute path or 'rel' for relative path.

    Returns:
    - str: Data path preferred.
    '''
    
    if sensor_type == 'abs':
        return '/workspaces/fitness-tracker/data/raw/MetaMotion/*.csv'
    elif sensor_type == 'rel':
        return '../../data/raw/MetaMotion/*.csv'
    else:
      raise Exception('Please, choose from `abs` or `rel` for absolute and relative path respectivelly!')

# Function to read and extract features from many raw data files

def data_reader(files:  Union[List[str], Tuple[str], Set[str]], verbose: bool = True) -> Tuple[pd.core.frame.DataFrame, pd.core.frame.DataFrame]:

  '''
  Reads the data files within a path and returns two DataFrames referring to the type of measurement made.
  It also extracts and handles relevant features from the file names. 
    
  Parameters:
  - files (Union[List[str], Tuple[str], Set[str]]):
    A collection of paths to data files in CSV format.
  - verbose (bool, optional):
      A boolean variable to indicate whether showing logs or not (default is `True`).
        Example: When setting to production, it's strongly recommended in order to keep track of everything.
    
  Returns:
  - Tuple[pd.core.frame.DataFrame]:
    A tuple containing two Pandas DataFrames:
        - The accelerometer data in the first DataFrame.
        - The gyroscope data in the second DataFrame.
        
  Notes:
  - Accelerometer x Gyroscope
    These are the two frequency of measurements.
    
    Acceleriometer -> 12.500Hz
    Gyroscope -> 25.000Hz
  '''
  
  if verbose:
    logger.info('Starting the empty DataFrames...')
  
  acc_df = pd.DataFrame()
  gyr_df = pd.DataFrame()

  acc_set, gyr_set = 1, 1

  for _ in files:
    df = pd.read_csv(_)
    
    _ = str.replace(_, '/workspaces/fitness-tracker', '../..')
    category = str.rstrip(str.rstrip(str.split(str.split(_, '-')[2], '_')[0], '123'), '_MetaWear_2019')
    participant = str.replace(str.split(_, '-')[0], get_raw_data_path('rel')[:-5], '')
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
      
  if verbose:
    logger.info('Standardizing the indexes...')
        
  acc_df.index = pd.Index(pd.to_datetime(acc_df['epoch (ms)'], unit='ms'))
  gyr_df.index = pd.Index(pd.to_datetime(gyr_df['epoch (ms)'], unit='ms'))

  if verbose:
    logger.info('Cleaning unnecessary time columns...')

  del acc_df['epoch (ms)']
  del acc_df['time (01:00)']
  del acc_df['elapsed (s)']

  del gyr_df['epoch (ms)']
  del gyr_df['time (01:00)']
  del gyr_df['elapsed (s)']

  if verbose:
    logger.success('The data has been read and the features were successfully extracted 🎉')
  
  return acc_df, gyr_df

# Function to subplot the X, Y, Z axis for each group within each subgroup
# - Usually used for each participant within each label

def axis_plots(df: pd.core.frame.DataFrame, group: str, subgroup: str, show: bool = False, verbose: bool = True) -> None:
  
  '''
  Generates subplots from a DataFrame with both a group and subgroup defined and saves them in a specific path.
  
  Parameters:
  - df (pd.core.frame.DataFrane):
    A Pandas DataFrame to extract the features to plot with.
    
  - group (str):
    A string containing the group inteded to use.
      Example: 'participant'.
    
  - subgroup (str):
    A string containing the subgroup intended to use.
      Example: 'label'.
    
  - show (bool):
    A boolean variable to indicate whether run `plt.show()` or not.
      Example: When using a Jupyter based notebook, you might want to include this option.
      
  - verbose (bool):
    A boolean variable to indicate whether showing logs or not.
      Example: When setting to production, it's strongly recommended in order to keep track of everything.
  '''
  
  if verbose:
    logger.info('Accessing the groups and subgroups...')
  
  group_ = sorted(df[f'{group}'].unique())
  subgroup_ = df[f'{subgroup}'].unique()
  
  for p in group_:
    if verbose:
      logger.info(f'Generating visualizations for group {p} and saving...')
      
    for l in subgroup_:
      plot_df = df.query(f"{subgroup} == '{l}'").query(f"{group} == '{p}'").reset_index()
    
      if len(plot_df) > 0:
        fig, ax = plt.subplots(nrows=2, sharex=True, figsize=(15, 10))

        plot_df[['acc_x', 'acc_y', 'acc_z']].plot(ax=ax[0])
        plot_df[['gyr_x', 'gyr_y', 'gyr_z']].plot(ax=ax[1])

        ax[0].legend(['X', 'Y', 'Z'], title='Accelerometer', bbox_to_anchor=(0.5, 1.18), loc='upper center', ncols=3, fancybox=True, shadow=True, frameon=False)
        ax[1].legend(['X', 'Y', 'Z'], title='Gyroscope', bbox_to_anchor=(0.5, 1.18), loc='upper center', ncols=3, fancybox=True, shadow=True, frameon=False)

        ax[0].set_xlabel('\nSample (time)\n')
        ax[0].set_ylabel('\nVariation\n')
        
        ax[1].set_xlabel('\nSample (time)\n')
        ax[1].set_ylabel('\nVariation\n')

        plt.suptitle(f'\nChanges for {l} in {group} {p}\n'.title())
        
        save_path = f'/workspaces/fitness-tracker/reports/figures/{str.capitalize(group)} {p} - {l.title()}.png'
        
        if os.path.exists(save_path):
          os.remove(save_path)
        
        plt.savefig(save_path, dpi=300)
        
        if show:
          plt.show()
        
        plt.close()
        
  if verbose:
    logger.success('The visualizations were made and successfully exported 🎉')