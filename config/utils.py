from sklearn.neighbors import LocalOutlierFactor
from config.config import settings
import matplotlib.pyplot as plt
from neptune.types import File
from datetime import datetime
from loguru import logger
import matplotlib as mpl
from io import StringIO
from io import BytesIO
import seaborn as sns
from glob import glob
from typing import *
import pandas as pd
import contextlib
import tempfile
import requests
import neptune
import scipy
import json
import math
import os

# Function to fetch API keys from a secret manager (Infisical)
# - Used for experiment tracking on Neptune

def secret_fetcher(query: str, env: str = 'dev', secrets_path: str = '/workspaces/fitness-tracker/.secrets.json', verbose: bool = True) -> Union[Tuple[str, str]]:
  
  '''
    Fetches a secret value from the Infisical (an encrypted platform for managing secrets) based on the provided query.
    
    PARAMETERS:
    
    - query (str, required): 
      A string referring to the identifier for the secret to retrieve.
      
    - env (str, required):
      A string referring to the environment to fetch the secret from (default is `dev`).
      
    - secrets_path (str, opional):
      A string referring to the file path to a JSON file containing API secrets (default is `/workspaces/fitness-tracker/.secrets.json`).
      
    - verbose (bool, optional):
      A boolean variable to indicate whether showing logs or not (default is `True`).
        Example: When setting to production, it's strongly recommended in order to keep track of everything.
  
    RETURNS:
    
    - Union[Tuple[str, str], None]: A tuple containing the secret key and value if successful, or None in case of an error.
    
    NOTES:
    
    - The function performs a POST request to acquire the API access key and then a GET request to fetch the desired secret.
  '''
  
  if query == '':
      raise Exception('Please, insert a valid query!')
  
  try:     
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
  except:
    logger.error('Internal error! Try again later.')
    raise SystemExit
  
# Function to get the raw data path 
# - This is a support for the next one - that'll be handling the data files
  
def get_raw_data_path(sensor_type: str) -> str:
  
    '''
    Get the data path based on the preference (whether absolute or relative).

    PARAMETERS:
    
    - sensor_type (str, required): 
      A string referring to the type of sensor.
        Options: 'abs' for absolute path or 'rel' for relative path.

    RETURNS:
    
    - str: Data path preferred.
    '''
    
    try:
      if sensor_type == 'abs':
          return '/workspaces/fitness-tracker/data/raw/MetaMotion/*.csv'
      elif sensor_type == 'rel':
          return '../../data/raw/MetaMotion/*.csv'
      else:
        raise Exception('Please, choose from `abs` or `rel` for absolute and relative path respectivelly!')
    except:
      logger.error('Internal error! Try again later.')
      raise SystemExit
      
# Function to read and extract features from many raw data files

def data_reader(files:  Union[List[str], Tuple[str], Set[str]], verbose: bool = True) -> Tuple[pd.core.frame.DataFrame, pd.core.frame.DataFrame]:

  '''
  Reads the data files within a path and returns two DataFrames referring to the type of measurement made.
  It also extracts and handles relevant features from the file names. 
    
  PARAMETERS:
  
  - files (Union[List[str], Tuple[str], Set[str]], required):
    A collection of paths to data files in CSV format.
    
  - verbose (bool, optional):
      A boolean variable to indicate whether showing logs or not (default is `True`).
        Example: When setting to production, it's strongly recommended in order to keep track of everything.
    
  RETURNS:
  
  - Tuple[pd.core.frame.DataFrame]:
    A tuple containing two Pandas DataFrames:
        - The accelerometer data in the first DataFrame.
        - The gyroscope data in the second DataFrame.
        
  NOTES:
  
  - Accelerometer x Gyroscope
    These are the two frequency of measurements.
    
    Acceleriometer -> 12.500Hz
    Gyroscope -> 25.000Hz
  '''
  
  try:
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
  except:
    logger.error('Internal error! Try again later.')
    raise SystemExit

# Function to subplot the X, Y, Z axis for each group within each subgroup
# - Usually used for each participant within each label

def axis_plotter(
    df: pd.core.frame.DataFrame, 
    group: str, 
    subgroup: str, 
    dark_theme: bool = False,
    color_palette: str = 'dark',
    show: bool = False, 
    verbose: bool = True
  ) -> None:
  
  '''
  Generates subplots from a DataFrame with both a group and subgroup defined and saves them in a specific path.
  
  PARAMETERS:
  
  - df (pd.core.frame.DataFrane, required):
    A Pandas DataFrame to extract the features to plot with.
    
  - group (str, required):
    A string containing the group inteded to use.
      Example: 'participant'.
    
  - subgroup (str, required):
    A string containing the subgroup intended to use.
      Example: 'label'.
      
  - dark_theme (bool, optional):
    A boolean variable to indicate whether plotting in dark theme or not (default is `False`).
    
  - color_pallete (str, optional):
    A string containing the color palette to use (default is `dark`).
      Example: `rocket_r`.
      
  - show (bool, optional):
    A boolean variable to indicate whether run `plt.show()` or not (default is `False`).
      Example: When using a Jupyter based notebook, you might want to include this option.
      
  - verbose (bool, optional):
    A boolean variable to indicate whether showing logs or not (default is `True`).
      Example: When setting to production, it's strongly recommended in order to keep track of everything.
      
  RETURNS:
  
  - None:
    Nothing is returned.
  '''
  
  try:
    _, api_token = secret_fetcher(query='NEPTUNE', env='dev', verbose=verbose)
  except TypeError as e:
    if verbose:
      logger.error('Internal error, review your query!')
      raise SystemExit
    else:
      raise SystemExit
    
  try:
      
    project_path = settings.project.experiment
    project = neptune.init_project(project=project_path, mode='async', api_token=api_token)
    
    project['general/brief'] = settings.project.brief
    
    id_ = datetime.now().strftime("%b %d, %Y @ %H:%M:%S")
    
    if verbose:
      run = neptune.init_run(
        custom_run_id='Axis Variation ' + datetime.now().strftime("%H:%M:%S"),
        name='Axis variations per participants within labels',
        description='Registering the visualizations on the axis of both sensors.',
        tags=['Report', 'Frequencies', 'Line'],
        source_files=[
          '/workspaces/fitness-tracker/config/utils.py',
          '/workspaces/fitness-tracker/src/visualization/visualize.py'
        ],
        git_ref=True,
        project=project_path,
        api_token=api_token,
        mode='async',
      )
    else:
      _ = StringIO()
      
      logger.disable('')
          
      with contextlib.redirect_stdout(_):
        with contextlib.redirect_stderr(_):
          with contextlib.redirect_stdout(_):
            run = neptune.init_run(
            custom_run_id='Axis Variation ' + datetime.now().strftime("%H:%M:%S"),
              name='Axis variations per participants within labels',
              description='Registering the visualizations on the axis of both sensors.',
              tags=['Report', 'Frequencies', 'Line'],
              source_files=[
                '/workspaces/fitness-tracker/config/utils.py',
                '/workspaces/fitness-tracker/src/visualization/visualize.py'
              ],
              git_ref=True,
              project=project_path,
              api_token=api_token,
              mode='async',
            )
            
    logger.enable('')
    
    if verbose:
      logger.info('Accessing the groups and subgroups...')  
      
    if dark_theme:
      plt.rcParams['figure.dpi'] = 300
      plt.style.use('dark_background')
      plt.rcParams['grid.color'] = '#212121'
      plt.rcParams['figure.max_open_warning'] = 30
    else:
      plt.rcdefaults()
      plt.rcParams['figure.dpi'] = 300
    
    sns.set_palette(color_palette)
    
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
              
          if verbose:    
            run[f'reports/figures/axis/{str.capitalize(group)} {p} - {l.title()}'].upload(fig)
            project[f'reports/axis/{str.capitalize(group)} {p} - {l.title()}'].upload(fig)
          else:
            logger.disable('')
                
            with contextlib.redirect_stdout(StringIO()):
              with contextlib.redirect_stderr(StringIO()):
                with contextlib.redirect_stdout(StringIO()):
                  run[f'reports/figures/axis/{str.capitalize(group)} {p} - {l.title()}'].upload(fig)
                  project[f'reports/axis/{str.capitalize(group)} {p} - {l.title()}'].upload(fig)
          
          logger.enable('')
          
          if show:
            plt.show()
          
          plt.close()
          
    run.stop()
          
    if verbose:
      logger.success('The visualizations were made and successfully exported 🎉')
  except:
    logger.error('Internal error! Try again later.')
    raise SystemExit