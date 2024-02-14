import sys
sys.path.append('/workspaces/fitness-tracker/')

from config.config import *
from config.utils import *

df = pd.read_pickle('/workspaces/fitness-tracker/data/interim/01_processed_data.pkl')

# ADD VERBOSITY HANDLE
def distribution_plotter(
    df: pd.core.frame.DataFrame, 
    sensor: str, 
    label: str,
    dark_theme: bool = False,
    color_palette: str = 'Set3',
    show: bool = False, 
    verbose: bool = True,
  ) -> None:
  
  '''
  Generates a plot from a DataFrame with the three axis distributions according to each label. 

  PARAMETERS:
  
  - df (pd.core.frame.DataFrane, required):
    A Pandas DataFrame to extract the features to plot with.
    
  - sensor (str, required):
    A string containing the sensor type intended to use.
      Example: `Accelerometer` or `Gyroscope`.
      
  - label (str, required):
    A string containing the categorical feature to compare by.
      Example: `category`.
      
  - dark_theme (bool, optional):
    A boolean variable to indicate whether plotting in dark theme or not (default is `False`).
      
  - color_pallete (str, optional):
    A string containing the color palette to use (default is `dark`).
      Example: `Set3`.
      
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
    
  project_path = settings.project.experiment
  project = neptune.init_project(project=project_path, mode='async', api_token=api_token)
  
  if sensor in ('Accelerometer', 'Gyroscope'):
    if sensor == 'Accelerometer':
      feature = 'acc_'
    else:
      feature = 'gyr_'
  else:
    raise Exception('Please, insert a valid sensor name!')
  
  if dark_theme:
      plt.rcParams['figure.dpi'] = 300
      plt.style.use('dark_background')
      plt.rcParams['grid.color'] = '#212121'
      plt.rcParams['figure.max_open_warning'] = 30
  else:
    plt.rcdefaults()
    plt.rcParams['figure.dpi'] = 300

  plt.figure(figsize=(18, 6))

  plt.suptitle(f'{str.capitalize(sensor)}\'s distributions by {str.capitalize(label)}', fontsize=16)

  plt.subplot(1, 3, 1)
  sns.boxplot(x=label, y=feature+'x', data=df, palette=color_palette, hue=label, flierprops=dict(marker='o', markersize=4))
  plt.xlabel(str.capitalize(label), fontsize=14)
  plt.ylabel('X-axis', fontsize=14)
  plt.xticks(rotation=45, ha='right')

  plt.subplot(1, 3, 2)
  sns.boxplot(x=label, y=feature+'y', data=df, palette=color_palette, hue=label, flierprops=dict(marker='o', markersize=4))
  plt.xlabel(str.capitalize(label), fontsize=14)
  plt.ylabel('Y-axis', fontsize=14)
  plt.xticks(rotation=45, ha='right')

  plt.subplot(1, 3, 3)
  sns.boxplot(x=label, y=feature+'z', data=df, palette=color_palette, hue=label, flierprops=dict(marker='o', markersize=4))
  plt.xlabel(str.capitalize(label), fontsize=14)
  plt.ylabel('Z-axis', fontsize=14)
  plt.xticks(rotation=45, ha='right')

  plt.tight_layout()
  
  buffer = BytesIO()
  plt.savefig(buffer, format='png')
  buffer.seek(0)
  
  with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tempfile_:
    tempfile_.write(buffer.read())
    tempfile_path = tempfile_.name
  
  if verbose:    
    project[f'reports/distributions/{sensor}'].upload(tempfile_path)
  else:
    logger.disable('')
        
    with contextlib.redirect_stdout(StringIO()):
      with contextlib.redirect_stderr(StringIO()):
        with contextlib.redirect_stdout(StringIO()):
          project[f'reports/distributions/{sensor}'].upload(tempfile_path)

  # plt.savefig()

  logger.enable('')
  
  if show:
    plt.show()



# def binary_outliers_plotter(
#     df: pd.core.frame.DataFrame, 
#     column: str, 
#     outlier_column: str, 
#     reset_index: bool = True,
#     dark_theme: bool = False,
#     color_palette: str = 'dark',
#     show: bool = False, 
#     verbose: bool = True
#   ) -> None:
  
#     ''' 
#     Generates a plot from a DataFrame with highlighted outliers in case of an IQR score.    
    
#     PARAMETERS:
    
#     - df (pd.core.frame.DataFrame, required):
#       A Pandas DataFrame to extract the features to plot with.
      
#     - column (str, required):
#       A string containing the column intended to plot with.
      
#     - outlier_column (str, required):
#       A string containing the outlier column marked with True/False.
      
#     - reset_index (bool, optional):
#       A boolean variable to indicate whether reset the index or not (default is `True`).
    
#     - dark_theme (bool, optional):
#       A boolean variable to indicate whether plotting in dark theme or not (default is `False`).
      
#     - color_pallete (str, optional):
#       A string containing the color palette to use (default is `dark`).
#         Example: `rocket_r`.
        
#     - show (bool, optional):
#       A boolean variable to indicate whether run `plt.show()` or not (default is `False`).
#         Example: When using a Jupyter based notebook, you might want to include this option.
        
#     - verbose (bool, optional):
#       A boolean variable to indicate whether showing logs or not (default is `True`).
#         Example: When setting to production, it's strongly recommended in order to keep track of everything.
        
#     RETURNS:
  
#     - None:
#       Nothing is returned.    
        
#     '''

#     df = df.dropna(axis=0, subset=[column, outlier_column])
#     df[outlier_column] = df[outlier_column].astype("bool")

#     if reset_index:
#       df = df.reset_index()
    
#     if dark_theme:
#       plt.rcParams['figure.dpi'] = 300
#       plt.style.use('dark_background')
#       plt.rcParams['grid.color'] = '#212121'
#       plt.rcParams['figure.max_open_warning'] = 30
#     else:
#       plt.rcdefaults()
#       plt.rcParams['figure.dpi'] = 300
      
#     sns.set_palette(color_palette)
    
#     fig, ax = plt.subplots(figsize=(10, 6))

#     # Plot non-outliers with adjusted marker size
#     ax.plot(
#         df.index[~df[outlier_column]],
#         df[column][~df[outlier_column]],
#         ".",  # Changed from "+" to "o" for a cleaner look
#         label="Non-Outliers",
#         markersize=3,
#         markerfacecolor='none',
#     )

#     # Plot data points that are outliers with adjusted marker size in red
#     ax.plot(
#         df.index[df[outlier_column]],
#         df[column][df[outlier_column]],
#         "o",  # Changed from "r+" to "ro" for consistency
#         color='#EC058E',
#         label="Outliers",
#         markersize=3,
#         markerfacecolor='none',
#     )
    
#     _ = 'Accelerometer' if outlier_column[:3] == 'acc' else 'Gyroscope'
    
#     plt.xlabel('Time')
#     plt.ylabel(f'{str.upper(outlier_column[:5][-1])}-axis')
    
#     plt.title(f'Outliers Detection for {column}')
#     plt.title(f'Outliers detection for {_}\'s {str.upper(outlier_column[:5][-1])}')

#     # Move the legend to the right outside the plt content
#     ax.legend(['non-outlier', 'outlier'], title=_, bbox_to_anchor=(1.01, 1.025), loc='upper left', ncols=1, fancybox=True, shadow=True, frameon=False)

#     ax.set_xticks(ax.get_xticks())
#     ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')  # Adjust rotation angle as needed

#     # Add grid lines for better readability
#     ax.grid(True, linestyle='--', alpha=0.7)
#     plt.show()
    
# def iqr_outliers_marker(df: pd.core.frame.DataFrame, column: str, verbose: bool = True) -> pd.core.frame.DataFrame:
    
#     '''
#     Adds a boolean column referring to the outliers' identification using the IQR method.
    
#     PARAMETERS:
    
#     - df (pd.core.frame.DataFrame, required):
#       A Pandas DataFrame to mark the outliers from.
      
#     - column (str, required):
#       A string containing the column name within the DataFrame.
      
#     - verbose (bool, optional):
#       A boolean variable to indicate whether showing logs or not (default is `True`).
#         Example: When setting to production, it's strongly recommended in order to keep track of everything.
      
#     RETURNS:
  
#     - pd.core.frame.DataFrame:
#       A Pandas DataFrame with the new column for the outliers marked
    
#     NOTES:
    
#     - The new column name is a pd.concat given by `column` + `_outlier`.
#     '''

#     df = df.copy()

#     if verbose:
#       logger.info('Getting the quantiles...')

#     Q1 = df[column].quantile(0.25)
#     Q3 = df[column].quantile(0.75)
    
#     if verbose:
#       logger.info('Getting the IQR value...')
    
#     IQR = Q3 - Q1

#     if verbose:
#       logger.info('Defining the bounds...')

#     lower_bound = Q1 - 1.5 * IQR
#     upper_bound = Q3 + 1.5 * IQR

#     if verbose:
#       logger.info('Generating the mask...')

#     df[column + "_outlier"] = (df[column] < lower_bound) | (
#         df[column] > upper_bound
#     )

#     if verbose:
#       logger.success('The outliers using the IQR method were successfully detected 🎉')

#     return df
  
# outlier_columns = df.columns[:6].tolist()

# column = 'acc_x'
# outlier_column = column+'_outlier'
# df = iqr_outliers_marker(df=df, column=column, verbose=False)

# binary_outliers_plotter(
#   df=df,
#   color_palette='rocket',
#   column=column,
#   dark_theme=True,
#   reset_index=True,
#   outlier_column=column+'_outlier',
#   verbose=True
# )

# for _ in outlier_column:
#   # df = mark_
#   pass