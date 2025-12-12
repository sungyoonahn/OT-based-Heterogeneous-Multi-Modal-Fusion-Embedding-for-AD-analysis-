import json
import random
import os
import argparse

def split_ids(input_file, output_file, val_fraction, seed):
    random.seed(seed)
    
    with open(input_file, 'r') as f:
        data = json.load(f)
        
    split_data = {"train": {}, "val": {}}
    
    for class_name, ids in data.items():
        # Ensure deterministic order before shuffling
        ids = sorted(ids)
        random.shuffle(ids)
        
        n_val = int(len(ids) * val_fraction)
        val_ids = ids[:n_val]
        train_ids = ids[n_val:]
        
        split_data["val"][class_name] = val_ids
        split_data["train"][class_name] = train_ids
        
        print(f"Class {class_name}: {len(train_ids)} train, {len(val_ids)} val")
        
    with open(output_file, 'w') as f:
        json.dump(split_data, f, indent=2)
    print(f"Saved fixed split to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    split_ids(args.input, args.output, args.val_fraction, args.seed)
