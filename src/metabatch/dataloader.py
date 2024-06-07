#! /usr/bin/env python3
# vim:fenc=utf-8
#
# Copyright © 2023 Théo Morales <theo.morales.fr@gmail.com>
#
# Distributed under terms of the MIT license.

"""
Meta-dataset dataloader.
"""


import random
from multiprocessing import Manager
from typing import (
    Callable,
    Iterator,
    List,
    Optional,
)

from torch.utils.data import (
    DataLoader,
    Dataset,
    RandomSampler,
    Sampler,
    SequentialSampler,
)

__all__ = ["TaskLoader"]


class SeededBatchSampler(Sampler):
    """
        Wraps another sampler to yield a mini-batch of indices, but seeds the dataset indexing to
        have consistant context + target set sizes across the batch.
        It is the default BatchSampler
        (https://pytorch.org/docs/stable/_modules/torch/utils/data/sampler.html#BatchSampler)
        modified to seed the context and target set sizes for all the elements of a batch
        (randomized at each batch).
    Args:
        dataset (Dataset): Dataset.
        batch_size (int): Size of mini-batch.
        drop_last (bool): If ``True``, the sampler will drop the last batch if
            its size would be less than ``batch_size``
    """

    def __init__(
        self,
        dataset,
        batch_size,
        drop_last,
        shuffle: bool = False,
        iterations: Optional[int] = None,
    ):
        if (
            not isinstance(batch_size, int)
            or isinstance(batch_size, bool)
            or batch_size <= 0
        ):
            raise ValueError(
                "batch_size should be a positive integer value, "
                "but got batch_size={}".format(batch_size)
            )
        if not isinstance(drop_last, bool):
            raise ValueError(
                "drop_last should be a boolean value, but got "
                "drop_last={}".format(drop_last)
            )
        if not isinstance(shuffle, bool):
            raise ValueError(
                "shuffle should be a boolean value, but got "
                "shuffle={}".format(shuffle)
            )
        self.batch_size = batch_size
        self.drop_last = drop_last if iterations is None else False
        self.n_batches = iterations
        self._sampling_inst = Manager().dict()
        dataset.register_sampling_inst(
            self._sampling_inst
        )  # Will modify every copy assigned to all workers, so they all get the same sampling_inst reference
        self.dataset = dataset
        if iterations is not None and iterations <= 0:
            raise ValueError(
                "Number of iterations should be a positive integer value, "
                "but got num_iterations={}".format(iterations)
            )
        if iterations is not None and not shuffle:
            raise ValueError(
                "shuffle should be True when num_iterations is not None, "
                "but got shuffle={}".format(shuffle)
            )
        self.sampler = (
            RandomSampler(
                dataset,
                replacement=iterations is not None,
                num_samples=iterations * batch_size if iterations is not None else None,
            )
            if shuffle
            else SequentialSampler(dataset)
        )

    def new_batch(self):
        n_context = random.randint(self.dataset.min_pts, self.dataset.max_ctx_pts)
        n_targets = (
            (
                self.dataset.total_tgt_pts - n_context
                if self.dataset.eval and self.dataset.predict_full_target_during_eval
                else random.randint(n_context, self.dataset.max_tgt_pts)
            )
            if not self.dataset.predict_full_target
            else self.dataset.total_tgt_pts
        )
        return n_context, n_targets

    def __iter__(self) -> Iterator[List[int]]:
        """
        The sampler gets a copy of the dataset and just returns indices for the dataloader. I
        modified the default BatchSampler to seed the context and target sizes of each new batch
        for all workers, using a multiprocessing dictionary.
        """
        # Implemented based on the benchmarking in https://github.com/pytorch/pytorch/pull/76951
        ctx_pts, tgt_pts = self.new_batch()
        if self.drop_last:
            sampler_iter = iter(self.sampler)
            while True:
                try:
                    batch = [next(sampler_iter) for _ in range(self.batch_size)]
                    for idx in batch:
                        self._sampling_inst[idx] = (ctx_pts, tgt_pts)
                    yield batch
                    ctx_pts, tgt_pts = self.new_batch()
                except StopIteration:
                    break
        else:
            batch = [0] * self.batch_size
            idx_in_batch = 0
            n_batches = 0
            if self.n_batches is not None:
                while n_batches < self.n_batches:  # type: ignore
                    for idx in self.sampler:
                        if n_batches >= self.n_batches:  # type: ignore
                            break
                        # Because I'm using a dict, I can't have duplicate indices in
                        # the batch. However, I'm setting replacement=True in the RandomSampler, so I can
                        # have duplicate indices. A quick hack is to skip the duplicate indices:
                        if idx in self._sampling_inst.keys():
                            # This effectively prevents the batch from having duplicate elements, while
                            # allowing to resample previously seen elements in future batches. This way, we
                            # can continue to train on the same dataset indefinitely or for an arbitrary
                            # number of iterations.
                            # We also need to update the number of samples in the sampler because it is not
                            # aware that we skipped some samples.
                            self.sampler._num_samples += 1  # type: ignore
                            continue
                        self._sampling_inst[idx] = (ctx_pts, tgt_pts)
                        batch[idx_in_batch] = idx
                        idx_in_batch += 1
                        if idx_in_batch == self.batch_size:
                            n_batches += 1
                            yield batch
                            idx_in_batch = 0
                            batch = [0] * self.batch_size
                            ctx_pts, tgt_pts = self.new_batch()
            else:
                for idx in self.sampler:
                    self._sampling_inst[idx] = (ctx_pts, tgt_pts)
                    batch[idx_in_batch] = idx
                    idx_in_batch += 1
                    if idx_in_batch == self.batch_size:
                        yield batch
                        idx_in_batch = 0
                        batch = [0] * self.batch_size
                        ctx_pts, tgt_pts = self.new_batch()
            if idx_in_batch > 0:
                yield batch[:idx_in_batch]

    def __len__(self):
        if self.n_batches is not None:
            return self.n_batches
        elif self.drop_last:
            return len(self.sampler) // self.batch_size
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size


class TaskLoader(DataLoader):
    def __init__(
        self,
        dataset: Dataset,
        batch_size: Optional[int] = 1,
        shuffle: bool = False,
        n_batches: Optional[int] = None,
        num_workers: int = 0,
        collate_fn: Optional[Callable] = None,
        pin_memory: bool = False,
        drop_last: bool = False,
        timeout: float = 0,
        worker_init_fn: Optional[Callable] = None,
        multiprocessing_context=None,
        generator=None,
        *,
        prefetch_factor: int = 2,
        persistent_workers: bool = True,
        pin_memory_device: str = ""
    ):
        super().__init__(
            dataset,
            batch_sampler=SeededBatchSampler(
                dataset, batch_size, drop_last, shuffle, n_batches
            ),
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=pin_memory,
            timeout=timeout,
            worker_init_fn=worker_init_fn,
            multiprocessing_context=multiprocessing_context,
            generator=generator,
            prefetch_factor=prefetch_factor,
            persistent_workers=persistent_workers,
            pin_memory_device=pin_memory_device,
        )
