import sys
sys.path.append('/workspaces/fitness-tracker/')

from config.utils import *

# Reading the processed data

df = pd.read_pickle('/workspaces/fitness-tracker/data/interim/01_processed_data.pkl')

# Generating and loading the subplots on X, Y, Z axis
# - For each participant within each label

axis_plotter(df=df, group='participant', subgroup='label', color_palette='dark', show=False, verbose=True)  