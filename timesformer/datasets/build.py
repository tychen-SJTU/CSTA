# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

from fvcore.common.registry import Registry

DATASET_REGISTRY = Registry("DATASET")
DATASET_REGISTRY.__doc__ = """
Registry for dataset.

The registered object will be called with `obj(cfg, split)`.
The call should return a `torch.utils.data.Dataset` object.
"""


def build_dataset(dataset_name, cfg, split):
    """
    Build a dataset, defined by `dataset_name`.
    Args:
        dataset_name (str): the name of the dataset to be constructed.
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        split (str): the split of the data loader. Options include `train`,
            `val`, and `test`.
    Returns:
        Dataset: a constructed dataset specified by dataset_name.
    """
    # Capitalize the the first letter of the dataset_name since the dataset_name
    # in configs may be in lowercase but the name of dataset class should always
    # start with an uppercase letter.
    name = dataset_name.capitalize()

    class_indexer = dict((i, n) for n, i in enumerate(cfg.CLASS_INDEXER))
    if split == 'train' or split == 'list':
        return DATASET_REGISTRY.get(name)(cfg.ROOT_PATH, cfg.TRAIN_LIST,
                                          cfg.CURRENT_TASK,
                                          class_indexer, split, cfg)
    elif split == 'balance':
        return DATASET_REGISTRY.get(name)(cfg.ROOT_PATH, cfg.BALANCE_LIST,
                                          cfg.TEST_TASK,
                                          class_indexer, split, cfg)
    else:
        return DATASET_REGISTRY.get(name)(cfg.ROOT_PATH, cfg.VAL_LIST,
                                          cfg.TEST_TASK,
                                          class_indexer, split, cfg)
