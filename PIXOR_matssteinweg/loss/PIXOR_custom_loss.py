import torch
import torch.nn as nn


##############
# focal loss #
##############
class FocalLoss(nn.Module):
    """
    Focal loss class. Stabilize training by reducing the weight of easily classified background sample and focussing
    on difficult foreground detections.
    """

    def __init__(self, gamma=0, size_average=False):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.size_average = size_average

    def forward(self, prediction, target):

        # get class probability
        pt = torch.where(target == 1.0, prediction, 1-prediction)

        # compute focal loss
        loss = -1 * (1-pt)**self.gamma * torch.log(pt)

        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()


######################
# Compute total loss #
######################
def compute_total_loss(batch_predictions, batch_labels):
    """
    Calculate the final loss function as a sum of the classification and the regression loss.
    :param batch_predictions: predictions for the current batch | shape: [batch_size, OUTPUT_DIM_0, OUTPUT_DIM_1, OUTPUT_DIM_CLA+OUTPUT_DIM_REG]
    :param batch_labels: labels for the current batch | shape: [batch_size, OUTPUT_DIM_0, OUTPUT_DIM_1, OUTPUT_DIM_CLA+OUTPUT_DIM_REG]
    :return: compouted loss
    """

    # classification loss
    classification_prediction = batch_predictions[:, :, :, -1].contiguous().flatten()
    classification_label = batch_labels[:, :, :, -1].contiguous().flatten()
    focal_loss = FocalLoss(gamma=2)
    classification_loss = focal_loss(classification_prediction, classification_label)

    # regression loss
    regression_prediction = batch_predictions[:, :, :, :-1]
    regression_prediction = regression_prediction.contiguous().view([regression_prediction.size(0)*
                        regression_prediction.size(1)*regression_prediction.size(2), regression_prediction.size(3)])
    regression_label = batch_labels[:, :, :, :-1]
    regression_label = regression_label.contiguous().view([regression_label.size(0)*regression_label.size(1)*
                                                           regression_label.size(2), regression_label.size(3)])
    positive_mask = torch.nonzero(torch.sum(torch.abs(regression_label), dim=1))
    pos_regression_label = regression_label[positive_mask.squeeze(), :]
    pos_regression_prediction = regression_prediction[positive_mask.squeeze(), :]
    smooth_l1 = nn.SmoothL1Loss(reduction='sum')
    regression_loss = smooth_l1(pos_regression_prediction, pos_regression_label)

    # add two loss components
    multi_task_loss = classification_loss.add(regression_loss)

    return multi_task_loss