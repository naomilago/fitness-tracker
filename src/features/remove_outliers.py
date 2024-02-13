import sys
sys.path.append('/workspaces/fitness-tracker/')

from config.utils import *

df = pd.read_pickle('/workspaces/fitness-tracker/data/interim/01_processed_data.pkl')

outlier_columns = df.columns[:6].tolist()

plt.rcParams['figure.dpi'] = 300

# plt.style.use('dark_background')
# plt.rcParams['grid.color'] = '#212121'
# plt.rcParams['figure.max_open_warning'] = 30

# Set up the subplots
plt.figure(figsize=(18, 6))

plt.suptitle('Acceleration\'s distributions by Exercises', fontsize=16)

plt.subplot(1, 3, 1)
sns.boxplot(x='label', y='acc_x', data=df, palette='Set3', hue='label', flierprops=dict(marker='o', markersize=4))
# plt.title('Acceleration by Exercises', fontsize=16)
plt.xlabel('Exercise', fontsize=14)
plt.ylabel('X-axis', fontsize=14)
plt.xticks(rotation=45, ha='right')

plt.subplot(1, 3, 2)
sns.boxplot(x='label', y='acc_y', data=df, palette='Set3', hue='label', flierprops=dict(marker='o', markersize=4))
# plt.title('Distribution of Acceleration Y by Label', fontsize=16)
plt.xlabel('Exercise', fontsize=14)
plt.ylabel('Y-axis', fontsize=14)
plt.xticks(rotation=45, ha='right')

plt.subplot(1, 3, 3)
sns.boxplot(x='label', y='acc_z', data=df, palette='Set3', hue='label', flierprops=dict(marker='o', markersize=4))
# plt.title('Distribution of Acceleration Z by Label', fontsize=16)
plt.xlabel('Exercise', fontsize=14)
plt.ylabel('Z-axis', fontsize=14)
plt.xticks(rotation=45, ha='right')

plt.tight_layout()
plt.show()

def plot_binary_outliers(dataset, column, outlier_column, reset_index: bool = True):
  
    ''' 
    Plots outliers in case of a binary outlier score.
    
    Parameters:
    
    - dataset (pd.core.frame.DataFrame, required):
      A Pandas DataFrame to extract the features to plot with.
      
    - column (str, required):
      A string containing the columnumn intended to plot with.
      
    - outier_column (str, required):
      A string containing the outlier column marked with True/False.
      
    - reset_index (str, optional):
      A boolean variable to indicate whether reset the index or not (default is `True`).
      
    - verbose (bool, optional):
      A boolean variable to indicate whether showing logs or not (default is `True`).
        Example: When setting to production, it's strongly recommended in order to keep track of everything.

    
    Plot outliers in case of a binary outlier score. Here, the column specifies the real data
    columnumn and outlier_column the columnumns with a binary value (outlier or not).

    Args:
        dataset (pd.DataFrame): The dataset
        column (string): columnumn that you want to plot
        outlier_column (string): Outlier columnumn marked with true/false
        reset_index (bool): whether to reset the index for plotting
    '''

    dataset = dataset.dropna(axis=0, subset=[column, outlier_column])
    dataset[outlier_column] = dataset[outlier_column].astype("bool")

    if reset_index:
        dataset = dataset.reset_index()

    fig, ax = plt.subplots()

    plt.xlabel("samples")
    plt.ylabel("value")

    # Plot non outliers in default color
    ax.plot(
        dataset.index[~dataset[outlier_col]],
        dataset[col][~dataset[outlier_col]],
        "+",
    )
    # Plot data points that are outliers in red
    ax.plot(
        dataset.index[dataset[outlier_col]],
        dataset[col][dataset[outlier_col]],
        "r+",
    )

    plt.legend(
        ["outlier " + col, "no outlier " + col],
        loc="upper center",
        ncol=2,
        fancybox=True,
        shadow=True,
    )
    plt.show()