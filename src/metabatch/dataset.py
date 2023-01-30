#! /usr/bin/env python3
# vim:fenc=utf-8
#
# Copyright © 2023 Théo Morales <theo.morales.fr@gmail.com>
#
# Distributed under terms of the MIT license.

"""
Meta-dataset base class.
"""


from multiprocessing.managers import DictProxy
from typing import List, Tuple, Union
from torch.utils.data import Dataset
import abc


class TaskSet(abc.ABC, Dataset):
    def __init__(
        self,
        min_pts: int,
        max_ctx_pts: int,
        max_tgt_pts: int,
        total_tgt_pts: int,
        eval: bool,
        predict_full_target: bool,
    ):
        self.max_ctx_pts = max_ctx_pts
        self.max_tgt_pts = max_tgt_pts
        self.min_pts = min_pts
        self.total_tgt_pts = total_tgt_pts
        self.eval = eval
        self.predict_full_target = predict_full_target
        self._sampling_instructor = None

    def register_sampling_inst(self, sampling_inst: DictProxy):
        self._sampling_instructor = sampling_inst

    def __getitem__(self, index: Union[int, List[int]]) -> Tuple:
        assert self._sampling_instructor is not None, "sampling_inst attribute not set!"
        try:
            n_context, n_target = self._sampling_instructor.pop(index)
        except KeyError as e:
            print(
                f"Could not get key {e} from sampling_inst -- make sure you are using the SeededBatchSampler as the batch_sampler argument of the DataLoader!"
            )
            raise e
        return self.__gettask__(index, n_context, n_target)


    @abc.abstractmethod
    def __gettask__(self, index, n_context, n_target):
        raise NotImplementedError
