# Introduction

MetaBatch provides convenient `Taskset` and `TaskLoader` classes for **batch-aware online task creation for meta-learning**.

## Efficient batching

Training meta-learning models efficiently can be a challenge, especially when it comes to creating
random tasks of a consistent shape in one batch. The task creation process can be time-consuming
and typically requires all tasks in the batch to have the same amount of context and target points.
This can be a bottleneck during training:

```python
# Sample code for creating a batch of tasks with traditional approach
class MyTaskDataset(Dataset):
    ...
    def __getitem__(self, idx):
        task = self.task_data[idx]
        return task

class Model(Module):
    ...
    def forward(self, tasks):
        ctx_batch = tasks['context']
        tgt_batch = tasks['target']
        ...

# create dataset
task_data = [{'images': [...], 'label': 'dog'},
             {'images': [...], 'label': 'cat'}, ...]
dataset = MyTaskDataset(task_data)
dataloader = DataLoader(dataset, batch_size=16, workers=8)

for batch in dataloader:
    ...
    # Construct batch of random tasks in the training loop (bottleneck!)
    n_context = random.randint(low=1, high=5)
    n_target = random.randint(low=1, high=10)
    tasks = {'context': [], 'target': []}
    for task in batch:
        context_images = sample_n_images(task['images'], n_context)
        target_images = sample_n_images(task['images'], n_target)
        tasks['context'].append(context_images)
        tasks['target'].append(target_images)
    model(tasks)
    ...
```

### Multiprocessing
Wouldn't it be better to offload the task creation to the dataloader, so that it can be done in
parallel on multiple cores?
With **MetaBatch**, we simplify the process by allowing you to do just that.
We provide a `TaskSet` wrapper, where you can implement the `__gettask__(self, index, n_context,
n_target)__` method instead of PyTorch's `__getitem(self, index)__`. Our `TaskLoader` and
custom sampler take care of synchronizing `n_context` and `n_target` for each batch element
dispatched to all workers. With **MetaBatch**, the training bottleneck can be removed from the
above example:
```python
# Sample code for creating a batch of tasks with MetaBatch
from metabatch import TaskSet, TaskLoader

class MyTaskSet(TaskSet):
    ...
    def __gettask__(self, idx, n_context, n_target):
        data = self.task_data[idx]
        context_images = sample_n_images(data['images'], n_context)
        target_images = sample_n_images(data['images'], n_target)
        return {
            'context': context_images
            'target': target_images
        }

class Model(Module):
    ...
    def forward(self, tasks):
        ctx_batch = tasks['context']
        tgt_batch = tasks['target']
        ...

# create dataset
task_data = [{'images': [...], 'label': 'dog'},
             {'images': [...], 'label': 'cat'}, ...]
dataset = MyTaskSet(task_data, min_pts=1, max_ctx_pts=5, max_tgt_pts=10)
dataloader = TaskLoader(dataset, batch_size=16, workers=8)

for batch in dataloader:
    ...
    # Simply access the batch of constructed tasks (no bottleneck!)
    model(batch)
    ...

```

## Installation & usage

Install it: `pip install metabatch`

Requirements:
- `pytorch`

Look at the example above for an idea or how to use `TaskLoader` with `TaskSet`, or go through the
examples in `examples/` (**TODO**).



## Advantages

- MetaBatch allows for efficient task creation and batching during training, resulting in more task
    variations since you are no longer limited to precomputed tasks.
- Reduces boilerplate needed to precompute and load tasks.

MetaBatch is a micro-framework for meta-learning in PyTorch that provides convenient tools for
(potentially faster) meta-training. It simplifies the task creation process and allows for efficient batching,
making it a useful tool for researchers and engineers working on meta-learning projects.

## How much faster?

**TODO**: benchmark MAML and CNP examples with typical implementation and other repos.


## License

MetaBatch is released under the MIT License. See the LICENSE file for more information.
