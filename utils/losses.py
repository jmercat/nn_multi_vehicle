import torch
import numpy as np

from utils.utils import Settings

args = Settings()

def simpleMSE_np(y_pred, y_gt, mask=None):
    y_pred_pos = y_pred.narrow(2, 0, 2)
    muX = y_pred_pos.narrow(2, 0, 1)
    muY = y_pred_pos.narrow(2, 1, 1)
    x = y_gt.narrow(2, 0, 1)
    y = y_gt.narrow(2, 1, 1)
    diff_x = x - muX
    diff_y = y - muY
    if mask is not None:
        output = np.sum((diff_x*diff_x + diff_y*diff_y)*mask)/np.sum(mask)
    else:
        output = np.mean(diff_x*diff_x + diff_y*diff_y)
    return output


def maskedMSE(y_pred, y_gt, mask=None, dim=3):
    muX = y_pred.narrow(dim, 0, 1)
    muY = y_pred.narrow(dim, 1, 1)
    x = y_gt.narrow(dim, 0, 1)
    y = y_gt.narrow(dim, 1, 1)
    diff_x = x - muX
    diff_y = y - muY
    if mask is not None:
        output = torch.sum((diff_x*diff_x + diff_y*diff_y)*mask.unsqueeze(dim))
        if torch.sum(mask) > 0:
            output = torch.sum(output)/torch.sum(mask)
        else:
            output = torch.sum(output)
    else:
        output = torch.mean(diff_x*diff_x + diff_y*diff_y)
    return output

