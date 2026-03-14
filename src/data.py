import yaml
from datasets import load_dataset

def load_config(path="configs/config.yaml"):
    with open(path) as f:
        config = yaml.safe_load(f)
    return config

def star_to_label(star):
    if star <= 1:
        return 0
    elif star == 2:
        return 1
    else:
        return 2

def load_review_data(cfg):
    ds = load_dataset(cfg['data']['dataset'], split='train')
    ds = ds.map(lambda x: {'label': star_to_label(x['label'])})
    ds = ds.train_test_split(test_size=cfg['data']['test_size'], seed=cfg['data']['seed'])
    return ds['train'], ds['test']