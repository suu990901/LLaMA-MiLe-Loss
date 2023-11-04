import os
import json
from tqdm import tqdm
import random
import sys
random.seed(42)

def merge_json_files(directory):
    merged_data = []
    for filename in tqdm(os.listdir(directory)):
        if filename.endswith(".jsonl"):
            with open(os.path.join(directory, filename), 'r') as f:
                data = f.readlines()
                merged_data.extend(data)
    
    return merged_data

# Partition the data for training in advance.
def divide_and_save(data, output_directory, num_shards=2048):
    chunk_size = len(data) // num_shards
    random.shuffle(data)
    for i in tqdm(range(num_shards)):
        start_idx = i * chunk_size
        end_idx = start_idx + chunk_size
        chunk = data[start_idx:end_idx]
        
        output_filename = os.path.join(output_directory, f'part_{i + 1}.jsonl')
        with open(output_filename, 'w') as f:
            f.writelines(chunk)

if __name__ == "__main__":
    split = sys.argv[1]
    data = sys.argv[2]
    input_root = "proc_pile_all"
    output_root = "proc_pile_2048"
    num_shards = 2048
    data = data.replace('_', ' ')
    input_directory = input_root + os.sep + split + os.sep + data
    output_directory = output_root + os.sep + split + os.sep + data
    os.makedirs(output_directory,exist_ok=True)
    merged_data = merge_json_files(input_directory)
    divide_and_save(merged_data, output_directory,num_shards)
    print("Merge and divide process completed.")