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
import os
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import transforms
from torchvision.datasets import MNIST
from tqdm import tqdm

from metabatch import TaskSet
from metabatch.dataloader import TaskLoader
import torch
from torch.nn import Module


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
        # TODO: Implement Attention or other fancy aggregators here
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


def plot_mean_picture(
    target_x,
    target_y,
    context_x,
    context_y,
    pred_y,
    var,
    img_dim,
    img_channels=3,
):
    """Plots the predicted mean and variance and the context points.

    Args:
      target_x: An array of shape batchsize x number_targets x 1 that contains the
          x values of the target pixels.
      target_y: An array of shape batchsize x number_targets x 3 that contains the
          R,G,B values of the target pixels.
      context_x: An array of shape batchsize x number_context x 1 that contains
          the x values of the context pixels.
      context_y: An array of shape batchsize x number_context x 3 that contains
          the R,G,B values of the context pixels.
      pred_y: An array of shape batchsize x number_targets x 3  that contains the
          predicted means of the R,G,B values at the target pixels in target_x.
      var: An array of shape batchsize x number_targets x 3  that contains the
          predicted variance of the R,G,B values at the target pixels in target_x.
    """
    # Plot side by side: top the context points on black background, middle the mean image prediction, bottom the standard deviation
    ctx = np.zeros((img_dim, img_dim, img_channels))
    target = np.zeros((img_dim, img_dim, img_channels))
    mean = np.zeros((img_dim, img_dim, img_channels))
    std_dev = np.zeros((img_dim, img_dim, img_channels))

    # Inputs are standardized inputs to [0,1]
    context_x = (context_x * (img_dim - 1)).type(torch.int)
    target_x = (target_x * (img_dim - 1)).type(torch.int)
    target = target_y.reshape(img_dim, img_dim, img_channels).swapdims(0, 1)
    # x is the coordinate for a vector image, y is (R,G,B) (or B,G,R ? Doesn't matter)
    for xy, rgb in zip(context_x, context_y):
        ctx[xy[0], xy[1]] = rgb
    for x, y in zip(target_x, target_y):
        target[int(x / img_dim), x % img_dim, :] = y
    for xy, rgb in zip(target_x, pred_y):
        mean[xy[0], xy[1]] = rgb
    for xy, rgb in zip(target_x, var):
        std_dev[xy[0], xy[1]] = rgb

    _, axarr = plt.subplots(4, 1)
    axarr[0].imshow(ctx, cmap="gray" if img_channels == 1 else "brg")
    axarr[0].set_ylabel(f"Context ({len(context_x)} pts)")
    axarr[1].imshow(target, cmap="gray" if img_channels == 1 else "brg")
    axarr[1].set_ylabel(f"Target ({len(target_x)} pts)")
    axarr[2].imshow(mean, cmap="gray" if img_channels == 1 else "brg")
    axarr[2].set_ylabel(f"Mean ({len(pred_y)} pts)")
    axarr[3].imshow(std_dev, cmap="gray" if img_channels == 1 else "brg")
    axarr[3].set_ylabel("Std. Dev.")
    plt.show()


def plot_picture_samples(
    target_x,
    target_y,
    context_x,
    context_y,
    pred,
    var,
    img_dim,
    img_channels=3,
):
    """Plots the predicted mean and variance and the context points.

    Args:
      target_x: An array of shape batchsize x number_targets x 1 that contains the
          x values of the target pixels.
      target_y: An array of shape batchsize x number_targets x 3 that contains the
          R,G,B values of the target pixels.
      context_x: An array of shape batchsize x number_context x 1 that contains
          the x values of the context pixels.
      context_y: An array of shape batchsize x number_context x 3 that contains
          the R,G,B values of the context pixels.
      pred: An array of shape batchsize x number_targets x 3  that contains sample functions
          for the R,G,B values at the target pixels in target_x.
      var: An array of shape batchsize x number_targets x 3  that contains the
          predicted variance of the R,G,B values at the target pixels in target_x.
    """
    # Plot side by side: top the context points on black background, middle the mean image prediction, bottom the standard deviation
    ctx = np.zeros((img_dim, img_dim, img_channels))
    target = np.zeros((img_dim, img_dim, img_channels))
    # std_dev = np.zeros((img_dim, img_dim, img_channels))

    # Inputs are standardized inputs to [0,1]
    context_x = (context_x * (img_dim - 1)).type(torch.int)
    target_x = (target_x * (img_dim - 1)).type(torch.int)
    target = target_y.reshape(img_dim, img_dim, img_channels).swapdims(0, 1)
    # x is the coordinate for a vector image, y is (R,G,B) (or B,G,R ? Doesn't matter)
    for xy, rgb in zip(context_x, context_y):
        ctx[xy[0], xy[1]] = rgb
    # for x, y in zip(target_x, target_y):
    # target[int(x / img_dim), x % img_dim, :] = y
    # for xy, rgb in zip(target_x, var):
    # std_dev[xy[0], xy[1]] = rgb

    _, axarr = plt.subplots(5, 1)
    axarr[0].imshow(ctx, cmap="gray" if img_channels == 1 else "brg")
    axarr[0].set_ylabel(f"Context ({len(context_x)} pts)")
    axarr[1].imshow(target, cmap="gray" if img_channels == 1 else "brg")
    axarr[1].set_ylabel(f"Target ({len(target_x)} pts)")
    # axarr[3, 0].imshow(std_dev, cmap="gray" if img_channels == 1 else "brg")
    # axarr[3, 0].set_ylabel("Std. Dev.")
    # Now plot 3 generated samples
    samples = [
        np.zeros((img_dim, img_dim, img_channels)),
        np.zeros((img_dim, img_dim, img_channels)),
        np.zeros((img_dim, img_dim, img_channels)),
    ]
    for i in range(3):
        if i >= pred.shape[0]:
            break
        for xy, rgb in zip(target_x, pred[i]):
            samples[i][xy[0], xy[1]] = rgb
        axarr[i + 2].imshow(samples[i], cmap="gray" if img_channels == 1 else "brg")
        axarr[i + 2].set_ylabel(f"Sample {i}")
    assert not np.allclose(samples[0], samples[1])
    assert not np.allclose(samples[1], samples[2])
    plt.show()


def main(
    model=None, train=True, vis=False, batch_size: int = 128, workers: int = 8
):

    epochs = 150
    val_every = 1
    img_size, img_channels = (MNISTDataset.IMG_SIZE[0], 1)
    max_ctx_pts = int(0.3 * (img_size**2))
    ckpt_path = f"ckpt"
    if not os.path.isdir(ckpt_path):
        os.makedirs(ckpt_path)

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

    ckpt = None
    if model is not None:
        print(f"[*] Loading model {model}...")
        ckpt = torch.load(model)
        cnp.load_state_dict(ckpt["model_ckpt"])

    if train:
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

    if train:
        min_val_loss = float("+inf")
        start_epoch = 0

        if ckpt is not None:
            opt.load_state_dict(ckpt["opt_ckpt"])
            scheduler.load_state_dict(ckpt["scheduler_ckpt"])
            start_epoch = ckpt["epoch"]
            min_val_loss = ckpt["val_loss"]

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
                    has_vised = False
                    val_loss, batches = 0.0, 0
                    for batch in tqdm(val_loader):
                        ctx_x, ctx_y, tgt_x, tgt_y = batch
                        dist, mu, var = cnp(ctx_x.cuda(), ctx_y.cuda(), tgt_x.cuda())
                        loss = -torch.mean(dist.log_prob(tgt_y.cuda()))
                        val_loss += loss.detach().item()
                        batches += 1
                        if not has_vised and vis:
                            for j in range(5):
                                plot_mean_picture(
                                    tgt_x[j],
                                    tgt_y[j],
                                    ctx_x[j],
                                    ctx_y[j],
                                    mu[j].cpu(),
                                    var[j].cpu(),
                                    img_size,
                                    img_channels=img_channels,
                                )
                            has_vised = True
                    val_loss /= batches
                    print(f"-> Validation loss: {val_loss}")
                    if val_loss < min_val_loss:
                        torch.save(
                            {
                                "model_ckpt": cnp.state_dict(),
                                "epoch": i,
                                "opt_ckpt": opt.state_dict(),
                                "scheduler_ckpt": scheduler.state_dict(),
                                "val_loss": val_loss,
                            },
                            os.path.join(
                                ckpt_path, f"epoch_{i}_val-loss_{val_loss:02f}.ckpt"
                            ),
                        )
                        min_val_loss = val_loss
            scheduler.step()
            print()

    if not train:
        with torch.no_grad():
            print("[*] Testing...")
            for batch in tqdm(val_loader):
                ctx_x, ctx_y, tgt_x, tgt_y = batch
                dist, mu, var = cnp(ctx_x.cuda(), ctx_y.cuda(), tgt_x.cuda())
                for j in range(3):
                    plot_mean_picture(
                        tgt_x[j],
                        tgt_y[j],
                        ctx_x[j],
                        ctx_y[j],
                        mu[j].cpu(),
                        var[j].cpu(),
                        img_size,
                        img_channels=img_channels,
                    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", dest="model", type=str, required=False, default=None)
    parser.add_argument("-t", dest="test", action="store_true", required=False)
    parser.add_argument(
        "-b",
        dest="batch_size",
        type=int,
        help="batch size",
        required=False,
        default=128,
    )
    parser.add_argument("--vis", action="store_true")
    parser.add_argument("-w", dest="workers", type=int, default=8, required=False)
    args = parser.parse_args()
    main(
        model=args.model,
        train=not args.test,
        vis=args.vis,
        batch_size=args.batch_size,
        workers=args.workers,
    )
