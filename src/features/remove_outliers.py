import sys
sys.path.append('/workspaces/fitness-tracker/')

from config.utils import *

df = pd.read_pickle('/workspaces/fitness-tracker/data/interim/01_processed_data.pkl')