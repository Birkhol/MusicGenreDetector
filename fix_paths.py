import pandas as pd
import os

# Define the old and new base paths
old_base = r'C:\VSCode Projects\MusicGenreDetector'
new_base = r'your-path-to-the-project-root'  # Update this to your actual path

# List of CSV files to update
csv_files = ['data/splits/train.csv', 'data/splits/val.csv', 'data/splits/test.csv']

for csv_file in csv_files:
    # Read the CSV
    df = pd.read_csv(csv_file)
    
    # Update the audio_path and image_path columns
    df['audio_path'] = df['audio_path'].str.replace(old_base, new_base, regex=False)
    df['image_path'] = df['image_path'].str.replace(old_base, new_base, regex=False)
    
    # Write back to CSV
    df.to_csv(csv_file, index=False)
    
    print(f"Updated {csv_file}")