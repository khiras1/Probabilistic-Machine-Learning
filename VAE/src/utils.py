from itertools import zip_longest
from typing import Any, Dict, List, Optional, Tuple, Union, Callable

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from sklearn.manifold import TSNE
from sklearn.metrics import mean_squared_error
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm, trange

from src.ae import BaseAutoEncoder


def train_ae(
    model: BaseAutoEncoder,
    epochs: int,
    train_loader: DataLoader,
    val_loader: DataLoader,
    lr: float,
    loss_fn: Callable,
    loss_fn_args: Optional[Tuple[Any]] = None,
    variational: bool = False,
) -> Tuple[Dict[str, List[float]], Dict[str, List[float]]]:
    """Train AE model and plot metrics.
    :param model: AE model
    :param epochs: number of epochs to train
    :param train_loader: train dataset loader
    :param val_loader: validation dataset loader
    :param lr: learning rate
    :param loss_fn: loss function to be applied
    :param loss_fn_args: optional args to be passed to loss function
        instead of input and output
    :return: train and val metrics
    """
    train_metrics, val_metrics = _initialize_metrics(variational)
    global_step = 0
    
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in trange(epochs, desc="epoch"):
        global_step = _train_epoch(
            model=model, 
            train_loader=train_loader, 
            optimizer=optimizer,
            loss_fn=loss_fn,
            loss_fn_args=loss_fn_args,
            variational=variational,
            train_metrics=train_metrics,
            global_step=global_step
        )
        
        _validate_model(
            model=model,
            val_loader=val_loader,
            loss_fn=loss_fn,
            loss_fn_args=loss_fn_args,
            variational=variational,
            val_metrics=val_metrics,
            global_step=global_step
        )

    return train_metrics, val_metrics


def _initialize_metrics(variational: bool = False) -> Tuple[Dict[str, List[float]], Dict[str, List[float]]]:
    """Initialize metrics dictionaries for training and validation."""
    train_metrics = {
        "loss": [],
        "mse": [],
        "step": [],
    }
    val_metrics = {
        "loss": [],
        "mse": [],
        "step": [],
    }
    if variational:
        train_metrics["kl"] = []
        val_metrics["kl"] = []
    return train_metrics, val_metrics


def _train_epoch(
    model: BaseAutoEncoder,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: Callable,
    loss_fn_args: Optional[Tuple[Any]],
    variational: bool,
    train_metrics: Dict[str, List[float]],
    global_step: int
) -> int:
    """Run one training epoch and update metrics.
    
    :param model: model to train
    :param train_loader: training data loader
    :param optimizer: optimizer instance
    :param loss_fn: loss function
    :param loss_fn_args: optional loss function arguments
    :param train_metrics: metrics dictionary to update
    :param global_step: current global step
    :return: updated global step
    """
    model.train()
    pbar = tqdm(train_loader, desc="step", leave=False)
    
    for inputs, _ in pbar:  # we are not using labels for training
        optimizer.zero_grad()
        reconstructions = model(inputs)
        
        if loss_fn_args is None:
            args = (reconstructions, inputs)
        else:
            args = (*loss_fn_args, reconstructions, inputs)
        
        loss = loss_fn(*args)
        loss.backward()
        optimizer.step()
        
        batch_size = inputs.shape[0]
        
        train_metrics["loss"].append(loss.item() / batch_size)
        train_metrics["step"].append(global_step)

        with torch.no_grad():
            mse = torch.nn.functional.mse_loss(
                input=inputs.view(batch_size, -1), 
                target=reconstructions.view(batch_size, -1), 
                reduction="sum"
            )
            train_metrics["mse"].append(mse.item() / batch_size)
        
            if variational:
                kl = model.kl_divergence().item()
                train_metrics["kl"].append(kl / batch_size)

        global_step += 1
        pbar.update(1)
        
    pbar.close()
    return global_step


def _validate_model(
    model: BaseAutoEncoder,
    val_loader: DataLoader,
    loss_fn: Callable,
    loss_fn_args: Optional[Tuple[Any]],
    variational: bool,
    val_metrics: Dict[str, List[float]],
    global_step: int
) -> None:
    """Validate model and update validation metrics.
    
    :param model: model to validate
    :param val_loader: validation data loader
    :param loss_fn: loss function
    :param loss_fn_args: optional loss function arguments
    :param val_metrics: metrics dictionary to update
    :param global_step: current global step
    """
    model.eval()
    val_loss = 0.0
    val_mse = 0.0
    val_kl = None if not variational else 0.0

    with torch.no_grad():
        total = 0
        for inputs, _ in val_loader:
            reconstructions = model(inputs)
            
            if loss_fn_args is None:
                args = (reconstructions, inputs)
            else:
                args = (*loss_fn_args, reconstructions, inputs)

            batch_size = inputs.shape[0]
            val_loss += loss_fn(*args) / batch_size
            val_mse += torch.nn.functional.mse_loss(
                input=inputs.view(batch_size, -1), 
                target=reconstructions.view(batch_size, -1), 
                reduction="sum"
            ) / batch_size
            if variational:
                val_kl += model.kl_divergence().item() / batch_size
            total += 1
        
    val_metrics["loss"].append(val_loss.item() / total)
    val_metrics["step"].append(global_step)
    val_metrics["mse"].append(val_mse.item() / total)
    if variational:
        val_metrics["kl"].append(val_kl / total)


def plot_metrics(train_metrics: Dict[str, List[float]], val_metrics: Dict[str, List[float]]):
    has_kl = "kl" in train_metrics
    fig_height = 12 if has_kl else 8
    n_plots = 3 if has_kl else 2
    
    fig, axes = plt.subplots(n_plots, 1, figsize=(5, fig_height), sharex=True)
    ax1, ax2 = axes[0], axes[1]
    ax3 = axes[2] if has_kl else None
    
    ax1.plot(train_metrics["step"], train_metrics["loss"], label="train loss")
    ax1.plot(val_metrics["step"], val_metrics["loss"], label="val loss")
    ax2.plot(train_metrics["step"], train_metrics["mse"], label="train mse")
    ax2.plot(val_metrics["step"], val_metrics["mse"], label="val mse")
    
    if has_kl:
        ax3.plot(train_metrics["step"], train_metrics["kl"], label="train kl")
        ax3.plot(val_metrics["step"], val_metrics["kl"], label="val kl")
        ax3.set_ylabel("KL")
        ax3.grid()
        ax3.legend()
    
    ax1.set_xlabel("Training step")
    ax1.set_ylabel("Loss")
    ax2.set_ylabel("MSE")
    ax1.set_title("Learning curves")
    
    for ax in [ax1, ax2]:
        ax.grid()
        ax.legend()
    
    plt.show()


class AutoEncoderAnalyzer:
    """Class for analysing an autoencoder model."""

    def __init__(
        self,
        model: BaseAutoEncoder,
        dataset: Dataset,
        n_samplings: int = 1,
    ):
        """
        :param model: trained autoencoder model
        :param dataset: test dataset
        :param n_samplings: number of samplings performed for analysis, defaults to 1
        """
        self.model = model
        self.dataset = dataset

        self.n_samplings = n_samplings

        self._inputs: Optional[torch.Tensor] = None
        self._latents: Optional[torch.Tensor] = None
        self._reconstructions: Optional[torch.Tensor] = None
        self._labels: Optional[torch.Tensor] = None

        self._plot_indices: Optional[np.array] = None

        self._class_indices: Dict[int, np.array] = {}
        self._averages: Optional[torch.Tensor] = None

        self._retrieve_reconstructions()

    def _retrieve_reconstructions(self):
        """Get data for analysis."""
        loader = DataLoader(self.dataset, batch_size=20, shuffle=False, drop_last=True)

        inps = []
        lats = []
        recs = []
        lbls = []

        for inputs, labels in loader:
            reconstructions = []
            latents = []
            for _ in range(self.n_samplings):
                latents.append(self.model.encoder_forward(inputs).detach())
                reconstructions.append(self.model(inputs).detach())
            inps.append(inputs)
            lats.append(torch.stack(latents, dim=1))
            recs.append(torch.stack(reconstructions, dim=1))
            lbls.append(labels)

        self._inputs = torch.cat(inps, dim=0).view(-1, 28, 28)
        self._latents = torch.cat(lats, dim=0).view(
            -1, self.n_samplings, self.model.n_latent_features
        )
        self._reconstructions = torch.cat(recs, dim=0).view(
            -1, self.n_samplings, 28, 28
        )
        self._labels = torch.cat(lbls, dim=0).view(-1)
        self._plot_indices = np.random.permutation(len(self._inputs))

    def compare_reconstruction_with_original(self):
        """Plot comparison between original image and its reconstruction."""
        indices = self._plot_indices[:50]
        inputs = self._inputs[indices]
        reconstructions = self._reconstructions[indices, 0, ...].unsqueeze(1)
        labels = self._labels[indices]

        visualize_samples(
            images=inputs,
            other_images=reconstructions,
            labels=labels.numpy(),
            n_cols=5,
            title="Inputs with reconstructions by AE",
        )

    def compare_samplings(self):
        """Plot comparison between samplings."""
        indices = self._plot_indices[:20]
        inputs = self._inputs[indices]
        reconstructions = self._reconstructions[indices]
        labels = self._labels[indices]

        visualize_samples(
            images=inputs,
            other_images=reconstructions,
            labels=labels.numpy(),
            n_cols=2,
            title="Inputs reconstructions - all samplings",
        )

    def average_points_per_class(self):
        """Calculate each class average points."""
        labels = self._labels.numpy()
        avgs = []
        for cls in np.sort(np.unique(labels)):
            cls_indices = np.where(labels == cls)
            self._class_indices[cls] = cls_indices
            class_latents = self._latents[list(cls_indices)].view(
                -1, self.model.n_latent_features
            )
            avgs.append(torch.mean(class_latents, dim=0))
        self._averages = torch.stack(avgs, dim=0)
        reconstructed = (
            self.model.decoder_forward(self._averages).detach().view(-1, 28, 28)
        )
        visualize_samples(
            images=reconstructed,
            labels=np.sort(np.unique(labels)),
            title="Reconstructed average representations per class",
        )

    def analyze_features(self, latent_code: torch.Tensor, steps: int = 11):
        """Perform latent feature analysis for a given latent code."""
        min_value = torch.floor(torch.min(self._latents)).item()
        max_value = torch.ceil(torch.max(self._latents)).item()
        print(f"Researching values in range [{min_value}, {max_value}]")
        latent_values = np.linspace(min_value, max_value, steps)
        reconstructed = []
        for idx in range(len(latent_code)):
            latents = []
            for latent_value in latent_values:
                new_latent = latent_code.clone()
                new_latent[idx] = latent_value
                latents.append(new_latent)
            feature_latents = torch.stack(latents, dim=0)
            feature_reconstructed = (
                self.model.decoder_forward(feature_latents).detach().view(-1, 28, 28)
            )
            reconstructed.append(feature_reconstructed)
        images = torch.stack(reconstructed, dim=0).view(-1, 28, 28)
        fig = visualize_samples(
            images=images, title="Latent features analysis", n_cols=steps
        )
        fig.text(
            0.5,
            0.0,
            f"latent feature value in range [{min_value}, {max_value}]",
            ha="center",
            va="center",
        )
        fig.text(
            0.0,
            0.5,
            "latent feature index",
            ha="center",
            va="center",
            rotation="vertical",
        )
        for feature_idx, axis in enumerate(fig.axes[::steps], 1):
            axis.set_ylabel(feature_idx)
        for feature_value_idx, axis in enumerate(fig.axes[-steps:]):
            axis.set_xlabel(round(latent_values[feature_value_idx], 1))

    def analyze_tsne(self):
        """Plot TSNE analysis of latent space."""
        tsne = TSNE(n_components=2)
        labels = self._labels.numpy()
        all_latents = torch.cat((self._latents[:, 0, :], self._averages), dim=0)
        embedded = tsne.fit_transform(all_latents.numpy())

        fig, ax = plt.subplots(figsize=(5, 5))
        for cls in np.sort(np.unique(labels)):
            class_embedded = embedded[self._class_indices[cls]]
            ax.scatter(class_embedded[:, 0], class_embedded[:, 1], label=cls, s=3)

        averages_embedded = embedded[
            len(labels) : len(labels) + len(self._class_indices)
        ]
        ax.scatter(
            averages_embedded[:, 0],
            averages_embedded[:, 1],
            s=50,
            c="k",
            marker="X",
        )

        ax.set_title("t-SNE analysis")
        ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), scatterpoints=10)


def visualize_samples(
    images: torch.Tensor,
    title: str,
    labels: Optional[Union[np.ndarray, List]] = None,
    other_images: Optional[torch.Tensor] = None,
    n_cols: int = 5,
):
    """Visualize images with their labels."""
    n_rows = len(images) // n_cols

    figsize = (n_cols, n_rows)
    if other_images is not None:
        figsize = (1 + other_images.shape[1]) * figsize[0], figsize[1]
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    axes = np.array(axes)

    if labels is None:
        labels = []
    if other_images is None:
        other_images = []

    for idx, (image, other, label) in enumerate(
        zip_longest(images, other_images, labels)
    ):
        x = idx % n_cols
        y = idx // n_cols
        ax = axes[y, x]
        if other is not None:
            for o in other:
                image = torch.cat((image, o), 1)
        ax.imshow(image.numpy(), cmap="gray")
        ax.text(0, 5, label, fontsize=12, weight="bold", c="w")
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(title, y=1)
    return fig
