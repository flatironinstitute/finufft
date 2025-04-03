import pandas as pd
import numpy as np

# Load the dataset
file_path = "wisdom.csv"  # Change this to the actual file path
df = pd.read_csv(file_path)

# Convert specified columns to numeric, coercing errors to NaN
columns_to_convert = ['Num_pts', 'Time_2.0', 'Time_1.25']
for col in columns_to_convert:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Drop rows with NaN values in these columns (optional, but recommended)
df = df.dropna(subset=columns_to_convert)
# scientific notation to Epsilon
df['Epsilon'] = df['Epsilon'].apply(lambda x: f'{x:.1e}')

# round Time_2.0 and Time_1.25 to 2 decimal places
df['Time_2.0'] = df['Time_2.0'].apply(lambda x: round(x, 2))
df['Time_1.25'] = df['Time_1.25'].apply(lambda x: round(x, 2))


# Extracting grid dimensions and volume from 'Size' column
def parse_grid_size(size):
    try:
        size = str(size).strip()
        if 'x' in size:
            dims = list(map(int, size.split('x')))
            volume = np.prod(dims)
            dim = len(dims)
        else:
            volume = int(size)
            dim = 1
        return volume, str(dim)
    except ValueError:
        return np.nan, np.nan


df[['Volume', 'Dim']] = df['Size'].apply(lambda x: pd.Series(parse_grid_size(str(x))))

# Compute density
df['Density'] = df['Num_pts'] / df['Volume']
# remove columns where Time_2.0 is > Time_1.25
df = df[df['Time_2.0'] <= df['Time_1.25']]

# drop columns where diff between Time_2.0 and Time_1.25 is less than 5%
df['% Diff'] = abs(df['Time_2.0'] - df['Time_1.25']) / df['Time_1.25']
df = df[df['% Diff'] >= .05]

# keep rows where n_threads is 1
df = df[df['n_threads'] == 16]
# drop time columns
df.drop(columns=['Time_2.0', 'Time_1.25', 'n_threads'], inplace=True)
df.drop(columns=['% Diff', 'Size', 'Num_pts', 'Volume', 'Unnamed: 0'], inplace=True)
# group by NUFFT_type, Data_type, Density, Dim, n_threads to make print more readable, concatenate values do not aggregate them
# Group by specific columns and concatenate values instead of aggregating them
grouped_df = df.groupby(['NUFFT_type', 'Data_type', 'Density', 'Dim'], as_index=False).agg(lambda x: ', '.join(x))
# grouped_df['Epsilon'] = grouped_df['Epsilon'].apply(lambda x: "{:.1e}".format(float(x)))

# sort by NUFFT type first, then Dim and then Density
grouped_df = grouped_df.sort_values(by=['NUFFT_type', 'Data_type', 'Dim', 'Density'])
grouped_df = grouped_df[['NUFFT_type', 'Data_type', 'Dim', 'Density', 'Epsilon']]

# Display the final dataset
print(grouped_df.to_string(index=False))
