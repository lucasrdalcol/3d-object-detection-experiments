#!/usr/bin/env python3

# import importlib
import torch.optim as optim
from PIXOR_matssteinweg.data_processing.load_data import *
from PIXOR_matssteinweg.models.PIXOR import PIXOR
import PIXOR_matssteinweg.utils.training_utils as training_utils
import PIXOR_matssteinweg.loss.PIXOR_custom_loss as custom_loss
import torch.nn as nn
import time
import numpy as np

# import wandb
sys.path.append(os.getenv("THREEDOBJECTDETECTION_ROOT"))
import PIXOR_matssteinweg.config.config as config
from PIXOR_matssteinweg.utils.common import *


# Seed for reproducibility
seed_everything(config.SEED)


###############
# train model #
###############
def train_model(
    model, optimizer, scheduler, data_loaders, n_epochs=25, show_times=False
):

    # evaluation dict
    metrics = {"train_loss": [], "val_loss": [], "lr": []}

    # early stopping object
    early_stopping = training_utils.EarlyStopping(
        patience=config.EARLY_STOPPING_PATIENCE, verbose=True
    )

    # moving loss
    if metrics["train_loss"] == []:
        moving_loss = {"train": None, "val": None}
    else:
        moving_loss = {
            "train": metrics["train_loss"][-1],
            "val": metrics["val_loss"][-1],
        }

    # epochs
    for epoch in range(n_epochs):

        # keep track of learning rate
        for param_group in optimizer.param_groups:
            metrics["lr"].append(param_group["lr"])

        # print header
        print("###########################################################")
        print("Epoch: " + str(epoch + 1) + "/" + str(n_epochs))
        print("Learning Rate: ", metrics["lr"][-1])

        # each epoch has a training and validation phase
        for phase in ["train", "val"]:
            print(f"phase: {phase}")

            # track average loss per batch
            if phase == "train":
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            for batch_id, (batch_data, batch_labels) in enumerate(data_loaders[phase]):
                # print(f"file index: {index}")

                # zero the parameter gradients
                optimizer.zero_grad()

                # track history only if in train phase
                with torch.set_grad_enabled(phase == "train"):

                    # forward pass
                    forward_pass_start_time = time.time()
                    batch_predictions = model(batch_data)
                    forward_pass_end_time = time.time()
                    batch_predictions = batch_predictions.permute([0, 2, 3, 1])

                    # calculate loss
                    calc_loss_start_time = time.time()
                    loss = custom_loss.compute_total_loss(
                        batch_predictions, batch_labels
                    )
                    calc_loss_end_time = time.time()

                    # accumulate loss
                    if moving_loss[phase] is None:
                        moving_loss[phase] = loss.item()
                    else:
                        moving_loss[phase] = (
                            0.99 * moving_loss[phase] + 0.01 * loss.item()
                        )

                    # append loss for each phase
                    metrics[phase + "_loss"].append(moving_loss[phase])

                    # backward + optimize only if in training phase
                    if phase == "train":
                        backprop_start_time = time.time()
                        loss.backward()
                        optimizer.step()
                        backprop_end_time = time.time()

                        if show_times:
                            print(
                                "Forward Pass Time: {:.2f}".format(
                                    forward_pass_end_time - forward_pass_start_time
                                )
                            )
                            print(
                                "Calc Loss Time: {:.2f} ".format(
                                    calc_loss_end_time - calc_loss_start_time
                                )
                            )
                            print(
                                "Backprop Time: {:.2f}".format(
                                    backprop_end_time - backprop_start_time
                                )
                            )

                    if (batch_id + 1) % 10 == 0:
                        n_batches_per_epoch = (
                            data_loaders[phase].dataset.__len__()
                            // data_loaders[phase].batch_size
                        )
                        print(
                            "{:d}/{:d} iterations | {} loss: {:.4f}".format(
                                batch_id + 1,
                                n_batches_per_epoch,
                                phase,
                                moving_loss[phase],
                            )
                        )

        # scheduler step
        scheduler.step()

        # output progress
        print("Training Loss: %.4f" % metrics["train_loss"][-1])
        print("Validation Loss: %.4f" % metrics["val_loss"][-1])

        # # Log metrics to wandb
        # wandb.log(
        #     {
        #         "epoch": epoch + 1,
        #         "train": {"loss": metrics['train_loss'][-1]},
        #         "val": {"loss": metrics['val_loss'][-1]},
        #     }
        # )

        # save metrics
        np.savez(
            os.path.join(
                config.RESULTS_DIR, "metrics/metrics_" + str(epoch + 1) + ".npz"
            ),
            history=metrics,
        )

        # check early stopping
        early_stopping(val_loss=metrics["val_loss"][-1], epoch=epoch, model=model)
        if early_stopping.early_stop:
            print("Early Stopping!")
            break

    print("Training Finished!")
    print(
        "Final Model was trained for "
        + str(early_stopping.best_epoch)
        + " epochs and achieved minimum loss of "
        "%.4f!" % early_stopping.val_loss_min
    )

    return metrics


def main():

    # # Config dict creation
    # spec = importlib.util.spec_from_file_location(
    #     "config", os.path.abspath(config.__file__)
    # )
    # config_module = importlib.util.module_from_spec(spec)
    # spec.loader.exec_module(config_module)
    # cfg = {
    #     name: getattr(config_module, name)
    #     for name in dir(config_module)
    #     if not name.startswith("__") and name.isupper()
    # }

    # # start a new wandb run to track this script
    # wandb.init(
    #     # set the wandb project where this run will be logged
    #     project="PIXOR",
    #     name=config.EXPERIMENT_NAME,
    #     # track hyperparameters and run metadata
    #     # config=cfg,
    #     # mode="disabled",
    # )

    # set device
    device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")

    # create data loader
    data_loader = load_dataset(
        root=config.DATASET_DIR,
        batch_size=config.BATCH_SIZE,
        device=device,
        train_val_split=config.TRAIN_VAL_SPLIT,
        num_workers=config.NUM_WORKERS,
        test_set=False,
    )

    # create model
    pixor = PIXOR().to(device)

    # create optimizer and scheduler objects
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, pixor.parameters()), lr=config.LEARNING_RATE
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [10, 20], gamma=0.1)

    # train model
    history = train_model(
        pixor, optimizer, scheduler, data_loader, n_epochs=config.N_EPOCHS
    )

    # save training history
    np.savez(os.path.join(config.RESULTS_DIR, "metrics/history.npz"), history=history)


if __name__ == "__main__":
    main()
