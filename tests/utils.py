#! /usr/bin/env python3
# vim:fenc=utf-8
#
# Copyright © 2023 Théo Morales <theo.morales.fr@gmail.com>
#
# Distributed under terms of the MIT license.

"""
Utility functions for testing.
"""

from torch.utils.data import default_collate

class ShapeMismatchError(Exception):
    def __init__(self, msg):
        self.msg = msg

def collate_verify_conformity(batch):
    """
    This function is to make sure the SeededBatchSampler functions as intended and that each worker
    yields samples with the same number of context/target points. It will check for any shape
    mismatch in the batch.
    """
    last_shape = None
    for el in batch:
        if last_shape is None:
            last_shape = el[0].shape
            continue
        if el[0].shape != last_shape:
            raise ShapeMismatchError(f"Mismatch: {el[0].shape} != {last_shape}")
    return default_collate(batch)
