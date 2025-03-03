# Imports
import os
import pandas as pd

# Functions
def generate_file_structure(start_path, indent=''):
  """
  Generate a file structure tree for a given directory path... to copy&paste on the README.md file...
  """
  file_structure = ''
  for item in os.listdir(start_path):
      if item == '.git':
        continue # Skip the .git folder
      
      item_path = os.path.join(start_path, item)
      if os.path.isdir(item_path):
        file_structure += f'{indent}├── {item}\n'
        file_structure += generate_file_structure(item_path, indent + '│   ')
      else:
        file_structure += f'{indent}├── {item}\n'
  return file_structure

def cols_overview(df):
    """
    It takes a DataFrame as input and returns a sorted DataFrame with the following columns:
    - nulls_count: number of missing values in the column.
    - col_name: name of the column.
    - col_dtype: data type of the values in the column.
    - nunique: number of unique values in the column.
    - unique: unique values in the column.
    - col_data_1: first 5 elements of the column.
    - col_data_2: last 5 elements of the column.
    
    Returns:
    - A DataFrame containing the overview of the columns in the input DataFrame.
    """
    cols = []
    for i in df:
        col = {'nulls_count': df[i].isnull().sum(),
            'col_name': i,
            'col_dtype': df[i].apply(type).value_counts(),
            'nunique': df[i].nunique(),
            'unique': df[i].unique(),
            'col_data_1': df[i].head(5).tolist(),
            'col_data_2': df[i].tail(5).tolist()}
        cols.append(col)
    to_df = pd.DataFrame(cols)
    sorted = to_df.sort_values(by='nulls_count', ascending=False)
    return sorted