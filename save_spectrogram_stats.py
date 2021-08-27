import numpy as np
import os
from params.params import Params as hp
import torch
import argparse
from collections import defaultdict
import pickle
import pdb



if __name__ == "__main__":
    # p = argparse.ArgumentParser()
    # p.add_argument("--checkpoint", type=str, required=True)
    # args = p.parse_args()

    stats_all = defaultdict(dict)
    checkpoint = "shared_initial5_simple_onehot/checkpoints/SHARED-TRAINING_loss-299-0.119-fisher"
    checkpoint_state = torch.load(checkpoint, map_location='cpu')
    hp.load_state_dict(checkpoint_state['parameters'])
    stats_all["initial5"]["mel_normalize_mean"] = hp.mel_normalize_mean
    stats_all["initial5"]["mel_normalize_variance"] = hp.mel_normalize_variance

    checkpoint_dir = "shared_initial5+continue5_simple_onehot_lr0.0005_ewc1.0/checkpoints"
    for lang in ["dutch", "finnish", "russian", "japanese", "greek"]:
        checkpoint = [ x for x in os.listdir(os.path.join(checkpoint_dir, lang))\
                        if x.endswith("fisher")][0]
        checkpoint_state = torch.load(os.path.join(checkpoint_dir, lang, checkpoint), map_location='cpu')
        hp.load_state_dict(checkpoint_state['parameters'])
        stats_all[lang]["mel_normalize_mean"] = hp.mel_normalize_mean
        stats_all[lang]["mel_normalize_variance"] = hp.mel_normalize_variance



    with open("stats_per_lang.pkl", "wb") as f:
        pickle.dump(stats_all, f)


