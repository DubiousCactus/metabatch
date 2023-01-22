#! /usr/bin/env python3
# vim:fenc=utf-8
#
# Copyright © 2023 Théo Morales <theo.morales.fr@gmail.com>
#
# Distributed under terms of the MIT license.

"""
Training a Conditional Neural Process (CNP) on MNIST with MetaBatch.
"""


import argparse
from typing import Tuple

import torch
from torch.nn import Module
from torchvision import transforms
from torchvision.datasets import MNIST
from tqdm import tqdm

from metabatch import TaskSet, TaskLoader


def make_MLP(n_layers, input_dim, output_dim, width):
    layers = [torch.nn.Linear(input_dim, width)]
    for _ in range(n_layers - 2):
        layers += [torch.nn.ReLU(inplace=True), torch.nn.Linear(width, width)]
    layers += [
        torch.nn.ReLU(inplace=True),
        torch.nn.Linear(width, output_dim),
    ]
    return torch.nn.Sequential(*layers)


class DeterministicEncoder(Module):
    def __init__(self, input_dim, output_dim, layers, width) -> None:
        super().__init__()
        self.mlp = make_MLP(
            n_layers=layers,
            input_dim=input_dim,
            output_dim=output_dim,
            width=width,
        )
        self._aggregator = lambda r: torch.mean(r, dim=1)

    def forward(self, ctx_x, ctx_y):
        context = torch.concat((ctx_x, ctx_y), dim=-1)  # Concat x and y
        r = self.mlp(context)
        return self._aggregator(r)


class Decoder(Module):
    def __init__(self, input_dim, output_dim, layers, width) -> None:
        super().__init__()
        self.mlp = make_MLP(
            n_layers=layers,
            input_dim=input_dim,
            output_dim=output_dim * 2,  # Same output dim for sigma and mu
            width=width,
        )
        self._output_dim = output_dim

    def forward(self, r_c, tgt_x):
        contextualised_targets = torch.concat(
            (
                r_c.unsqueeze(1).expand((-1, tgt_x.shape[1], -1)),
                tgt_x,
            ),
            dim=-1,
        )
        output = self.mlp(contextualised_targets)
        mu, log_sigma = (
            output[..., : self._output_dim],
            output[..., self._output_dim :],
        )
        sigma = 0.1 + 0.9 * torch.nn.functional.softplus(log_sigma)
        # Independent allows to "Reinterprets some of the batch dims of a distribution as event dims".
        # That's really useful because we have a tensor of (BATCH_SIZE, POINTS_PER_SAMPLE, POINT_DIM),
        # where we want POINTS_PER_SAMPLE (our events) to be distributed by independent normals (dim=1)!
        return (
            torch.distributions.Independent(torch.distributions.Normal(mu, sigma), 1),
            mu,
            sigma,
        )


class CNP(Module):
    def __init__(
        self,
        encoder_input_dim,
        decoder_input_dim,
        output_dim,
        encoder_dim=128,
        decoder_dim=128,
        encoder_layers=4,
        decoder_layers=3,
    ) -> None:
        super().__init__()
        # The encoder (MLP) takes in pairs of (x, y) context points and returns r_i
        #   -> It concatenates x and y
        #   -> Num of layers is a hyperparameter (4?)
        #   -> Width of layers is a hyperparameter (4?)
        #   -> ReLU activations except for the last layer
        self.encoder = DeterministicEncoder(
            input_dim=encoder_input_dim,
            output_dim=encoder_dim,
            layers=encoder_layers,
            width=encoder_dim,
        )
        #   -> Of dim r_i
        # The decoder (MLP) takes in the aggregated r, a latent variable z, and a target x_i to
        # produce a mean estimate mu_i + sigma_i
        #   -> Num of layers is a hyperparameter (4?)
        #   -> Width of layers is a hyperparameter (4?)
        self.decoder = Decoder(
            encoder_dim + decoder_input_dim, output_dim, decoder_layers, decoder_dim
        )

    def forward(self, ctx_x, ctx_y, tgt_x):
        """
        A batch is a batch of function samples with the same amount of context and target points.
        Context: [(BATCH_SIZE, N_CONTEXT_PTS, DATA_DIM), (BATCH_SIZE, N_CONTEXT_PTS, DATA_DIM)]
        Targets: (BATCH_SIZE, N_TARGET_PTS, DATA_DIM)
        """
        # Encode the input/output pairs (one pair per context point) and aggregate them into a mean
        # encoded context input/output pair.
        r_c = self.encoder(ctx_x, ctx_y)
        # Decode each pair of [mean encoded context input/output pair, target input] into a target
        # output.
        return self.decoder(r_c, tgt_x)


class MNISTDataset(TaskSet):
    IMG_SIZE = (28, 28)

    def __init__(self, min_ctx_pts, max_ctx_pts, eval) -> None:
        super().__init__(
            min_ctx_pts,
            max_ctx_pts,
            self.IMG_SIZE[0] ** 2,
            self.IMG_SIZE[0] ** 2,
            eval,
            predict_full_target=False,
        )
        print(
            f"[*] Loading {'training' if not eval else 'validation'} MNIST data set..."
        )
        self._img_dim = self.IMG_SIZE[0]
        self._training_samples = self._load_samples(eval)

    def _load_samples(self, eval):
        return MNIST(
            "data",
            train=not eval,
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Lambda(
                        lambda x: torch.moveaxis(
                            x, 0, 2
                        )  # Channel first to channel last
                    ),
                ]
            ),
            download=True,
        )

    def __gettask__(self, index: int, n_context: int, n_target: int) -> Tuple:
        """ "
        We're not gonna use n_target because we're predicting the full target.
        """
        img = self._training_samples[index][0]
        ctx_x = torch.randint(0, self._img_dim, size=(n_context, 2))
        # This is probably the most efficient way to sample 2D coordinates, but it'll have
        # duplicates! Is it really a problem? I doubt it (the result is effectively less context
        # points, but since we're continuously sampling n_context it doesn't matter!)
        ctx_y = img[ctx_x[:, 0], ctx_x[:, 1]]

        # The targets should not have any duplicates!!
        # This is for the whole image as target:
        axis = torch.arange(0, self._img_dim)
        tgt_x = torch.stack(torch.meshgrid(axis, axis, indexing="xy"), dim=2).reshape(
            self._img_dim**2, 2
        )
        tgt_y = img[tgt_x[:, 0], tgt_x[:, 1]]
        # Standardize inputs to [0,1]
        ctx_x = ctx_x.type(torch.float32) / (self._img_dim - 1)
        tgt_x = tgt_x.type(torch.float32) / (self._img_dim - 1)
        return ctx_x, ctx_y, tgt_x, tgt_y

    def __len__(self) -> int:
        assert self._training_samples is not None, "training_samples attribute not set!"
        return len(self._training_samples)


def main(batch_size: int = 128, workers: int = 8):

    epochs = 150
    val_every = 1
    img_size, img_channels = (MNISTDataset.IMG_SIZE[0], 1)
    max_ctx_pts = int(0.3 * (img_size**2))

    print(f"[*] Using max {max_ctx_pts} context points")
    # Encoder input dim = 1 (coord) + 1 (channels) = 4
    # Decoder output dim = 1 (mu) + 1 (sigma) = 2
    cnp = CNP(
        encoder_input_dim=2 + img_channels,
        decoder_input_dim=2,
        output_dim=img_channels,
        encoder_dim=128,
        decoder_dim=128,
        encoder_layers=3,
        decoder_layers=5,
    ).cuda()
    opt = torch.optim.Adam(cnp.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=30, gamma=0.5)

    train_dataset = MNISTDataset(
        min_ctx_pts=5,
        max_ctx_pts=max_ctx_pts,
        eval=False,
    )
    train_loader = TaskLoader(
        train_dataset,
        num_workers=workers,
        batch_size=batch_size,
        shuffle=True,
    )

    val_dataset = MNISTDataset(
        min_ctx_pts=5,
        max_ctx_pts=max_ctx_pts,
        eval=True,
    )
    val_loader = TaskLoader(
        val_dataset,
        num_workers=workers,
        batch_size=batch_size,
        shuffle=False,
    )

    start_epoch = 0

    for i in range(start_epoch, epochs):
        print(f"Epoch {i+1}/{epochs}")
        epoch_loss, batches = 0.0, 0
        for batch in tqdm(train_loader):
            opt.zero_grad()
            ctx_x, ctx_y, tgt_x, tgt_y = batch
            dist, _, _ = cnp(ctx_x.cuda(), ctx_y.cuda(), tgt_x.cuda())
            loss = -torch.mean(dist.log_prob(tgt_y.cuda()))
            epoch_loss += loss.detach().item()
            batches += 1
            loss.backward()
            opt.step()
        epoch_loss /= batches
        print(f"--> Loss={epoch_loss}")

        if i % val_every == 0:
            with torch.no_grad():
                val_loss, batches = 0.0, 0
                for batch in tqdm(val_loader):
                    ctx_x, ctx_y, tgt_x, tgt_y = batch
                    dist, mu, var = cnp(ctx_x.cuda(), ctx_y.cuda(), tgt_x.cuda())
                    loss = -torch.mean(dist.log_prob(tgt_y.cuda()))
                    val_loss += loss.detach().item()
                    batches += 1
                val_loss /= batches
                print(f"-> Validation loss: {val_loss}")
        scheduler.step()
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-b",
        dest="batch_size",
        type=int,
        required=False,
        default=128,
    )
    parser.add_argument("-w", dest="workers", type=int, default=8, required=False)
    args = parser.parse_args()
    main(
        batch_size=args.batch_size,
        workers=args.workers,
    )
