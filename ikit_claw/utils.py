import pickle
import numpy as np

def remove_endline(stringy: str):
    return stringy[:-1] if stringy.endswith("\n") else stringy

def load_pickle(path):
    with open(path, 'rb') as f:
        ret = pickle.load(f)
    return ret

def split_container_by_ratio(data, split_ratio, seed=2667):
    """
    Split a python container e.g. list, dataset, into random splits
        - data = python container
        - split_ratio = (1,2,3) as a tuple of ratios to split
        - seed = random fixable seed
    """
    # Fixed random shuffling
    rng = np.random.Generator(np.random.PCG64(seed))
    rng.shuffle(data)
    splits = []
    total = sum(split_ratio)
    for idx in range(len(split_ratio)):
        start = sum(split_ratio[:idx])/total
        start = round(start*len(data))
        end = sum(split_ratio[:idx+1])/total
        end = round(end*len(data))
        splits.append(data[start:end])
    return splits

