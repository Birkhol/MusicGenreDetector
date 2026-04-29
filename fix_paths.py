import pandas as pd
import os

# Update this to your actual path
new_base = r'change-to-your-root-path'

# List of CSV files to update
csv_files = ['data/splits/train.csv', 'data/splits/val.csv', 'data/splits/test.csv']

for csv_file in csv_files:
    # Read the CSV
    df = pd.read_csv(csv_file)
    
    # This regex finds the "data" folder and replaces everything before it with the new_base
    # regex=True is required to use the '^.*(?=data)' pattern
    df['audio_path'] = df['audio_path'].str.replace(r'^.*(?=data)', new_base + '/', regex=True)
    df['image_path'] = df['image_path'].str.replace(r'^.*(?=data)', new_base + '/', regex=True)
    
    # Write back to CSV
    df.to_csv(csv_file, index=False)
    
    print(f"Updated {csv_file}")