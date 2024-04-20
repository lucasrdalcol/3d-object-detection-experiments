#!/usr/bin/env python3

import os
import numpy as np
import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import sys

sys.path.append(os.getenv("THREEDOBJECTDETECTION_ROOT"))
import PIXOR_matssteinweg.config.config as config


def plot_history(metrics, n_epochs_trained, save=True, show=False):
    """
    Plot evoluction of training and validation loss over the training period.
    :param metrics: dictionary containing training and validation loss
    """

    fig, axs = plt.subplots(1, 2, figsize=(25, 10))
    plt.subplots_adjust(top=0.75, bottom=0.25, wspace=0.4)

    train_loss = metrics["train_loss"]
    val_loss = metrics["val_loss"]

    # batch_size = 2
    # epochs = batch_size * (len(train_loss) + len(val_loss)) // 6481

    for ax_id, ax in enumerate(axs):
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Loss")
        ax.set_yscale("log")
        ax.grid(True)

        if ax_id == 0:
            ax.set_ylim([0.5 * min(train_loss), max(train_loss)])
            ax.set_xlim([0.0, len(train_loss)])
            ax.set_title("Training Loss")
            step_size = 3
            ticks = np.arange(
                0, len(train_loss), step_size * len(train_loss) // (n_epochs_trained)
            )
            labels = np.arange(1, n_epochs_trained + 1, step_size)
            ax.set_xticks(ticks)
            ax.set_xticklabels(labels)
            ax.plot(train_loss)
        else:
            ax.set_ylim([0.5 * min(val_loss), max(val_loss)])
            ax.set_xlim([0.0, len(val_loss)])
            ax.set_title("Validation Loss")
            step_size = 3
            ticks = np.arange(
                0, len(val_loss), step_size * len(val_loss) // (n_epochs_trained)
            )
            labels = np.arange(1, n_epochs_trained + 1, step_size)
            ax.set_xticks(ticks)
            ax.set_xticklabels(labels)
            ax.plot(val_loss)

    if save:
        plt.savefig(
            os.path.join(
                config.RESULTS_DIR, "metrics/training_validation_loss_per_epochs.png"
            ),
            dpi=300,
            bbox_inches="tight",
        )
    if show:
        plt.show()


def main():
    metrics = np.load(
        os.path.join(config.RESULTS_DIR, f"metrics/metrics_{config.N_EPOCHS_TRAINED}.npz"),
        allow_pickle=True,
    )["history"].item()
    plot_history(metrics, config.N_EPOCHS_TRAINED, save=True, show=False)


if __name__ == "__main__":
    main()
