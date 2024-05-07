import pandas as pd
import shutil
import os
from rich import print
from tqdm import tqdm

split = 'val'
df = pd.read_csv(f'{split}_data.csv')
label_list = sorted(df['label'].unique())

# Create a folder for each label
for label in label_list:
    target_folder = f'{split}/{label}/'
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

target_folder = f'{split}/'
if not os.path.exists(target_folder):
    os.makedirs(target_folder)


counter = 0
for index, row in tqdm(df.iterrows()):
    counter += 1
    image_name = row['image_name']
    label = row['label']

    source_path = os.path.join("imgs", image_name)
    target_path = os.path.join(target_folder, str(label),  image_name)

    os.system("cp {} {}".format(source_path, target_path))
    # print(f'Moved {image_name} to {target_folder}')

print(counter)
