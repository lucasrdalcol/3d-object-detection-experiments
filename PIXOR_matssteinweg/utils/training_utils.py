import torch
import numpy as np
import sys
import os

sys.path.append(os.getenv("THREEDOBJECTDETECTION_ROOT"))
import PIXOR_matssteinweg.config.config as config


##################
# early stopping #
##################
class EarlyStopping:
    """
    Early stops the training if validation loss doesn't improve after a given patience.
    """

    def __init__(self, patience=7, verbose=False):
        """
        :param patience: How many epochs wait after the last validation loss improvement
        :param verbose: If True, prints a message for each validation loss improvement.
        """

        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.best_epoch = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, val_loss, epoch, model):

        score = -val_loss

        # first epoch
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch + 1
            self.save_checkpoint(val_loss, model)

        # validation loss increased
        elif score < self.best_score:

            # increase counter
            self.counter += 1

            print(
                "Validation loss did not decrease ({:.6f} --> {:.6f})".format(
                    self.val_loss_min, val_loss
                )
            )
            print(
                "EarlyStopping counter: {} out of {}".format(
                    self.counter, self.patience
                )
            )
            print("###########################################################")

            # stop training if patience is reached
            if self.counter >= self.patience:
                self.early_stop = True

        # validation loss decreased
        else:
            self.best_score = score
            self.best_epoch = epoch + 1
            self.save_checkpoint(val_loss, model)

            # reset counter
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """
        Saves model when validation loss decreased.
        """

        if self.verbose:
            print(
                "Validation loss decreased ({:.6f} --> {:.6f}).  "
                "Saving model ...".format(self.val_loss_min, val_loss)
            )
            print("###########################################################")

        # save model
        torch.save(
            model.state_dict(),
            os.path.join(config.MODELS_DIR, "PIXOR_Epoch_")
            + str(self.best_epoch)
            + ".pt",
        )

        # set current loss as new minimum loss
        self.val_loss_min = val_loss
