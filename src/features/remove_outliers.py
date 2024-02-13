import sys
sys.path.append('/workspaces/fitness-tracker/')

from config.utils import *

df = pd.read_pickle('/workspaces/fitness-tracker/data/interim/01_processed_data.pkl')

outlier_columns = df.columns[:6].tolist()

plt.rcParams['figure.dpi'] = 300

# plt.style.use('dark_background')
# plt.rcParams['grid.color'] = '#212121'
# plt.rcParams['figure.max_open_warning'] = 30

sns.set_palette('muted')

# Create a beautiful boxplot with smaller outliers
plt.figure(figsize=(14, 8))
sns.boxplot(x='label', y='acc_x', data=df, palette='Set3', hue='label', flierprops=dict(marker='o', markersize=4))
plt.title('Distribution of Acceleration X by Label', fontsize=16)
plt.xlabel('Exercise label', fontsize=14)
plt.ylabel('Acceleration X', fontsize=14)
plt.xticks(rotation=45, ha='right')
sns.despine(trim=True, left=True)

# Display the plot
plt.tight_layout()
plt.show()


# Set up the subplots
plt.figure(figsize=(18, 6))

# Subplot 1
plt.subplot(1, 3, 1)
sns.boxplot(x='label', y='acc_x', data=df, palette='Set3', hue='label', flierprops=dict(marker='o', markersize=4))
plt.title('Distribution of Acceleration X by Label', fontsize=16)
plt.xlabel('Exercise label', fontsize=14)
plt.ylabel('Acceleration X', fontsize=14)
plt.xticks(rotation=45, ha='right')

# Subplot 2
plt.subplot(1, 3, 2)
sns.boxplot(x='label', y='acc_y', data=df, palette='Set3', hue='label', flierprops=dict(marker='o', markersize=4))
plt.title('Distribution of Acceleration Y by Label', fontsize=16)
plt.xlabel('Exercise label', fontsize=14)
plt.ylabel('Acceleration Y', fontsize=14)
plt.xticks(rotation=45, ha='right')

# Subplot 3
plt.subplot(1, 3, 3)
sns.boxplot(x='label', y='acc_z', data=df, palette='Set3', hue='label', flierprops=dict(marker='o', markersize=4))
plt.title('Distribution of Acceleration Z by Label', fontsize=16)
plt.xlabel('Exercise label', fontsize=14)
plt.ylabel('Acceleration Z', fontsize=14)
plt.xticks(rotation=45, ha='right')

# Adjust layout and display the plot
plt.tight_layout()
plt.show()