"""Mixture/replicate group IDs: rows sharing the same acquisition date and
EVOO% are the same prepared mixture's 5 replicate spectra. Used to make
every internal cross-validation grouped, so replicates of one mixture can
no longer land in both the training and validation side of a CV fold."""
import numpy as np


def get_groups(meta):
    key = meta["date"].astype(str) + "|" + meta["evoo_pct"].astype(str)
    # encode as integer codes for sklearn's GroupKFold
    _, codes = np.unique(key.to_numpy(), return_inverse=True)
    return codes
