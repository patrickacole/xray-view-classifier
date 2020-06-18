import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

def accuracy_calculation(predicts, labels, threshold=0.5):
    """
    Calculate accuracy 0%-100%
    @param  predicts: predictions from the network (Tensor)
    @param  labels  : actual labels (Tensor)
    @return accuracy: number of correct divided by total
    """
    zero = torch.zeros(1)
    one = torch.ones(1)
    predicted_labels = torch.sigmoid(predicts)
    predicted_labels = torch.where(predicted_labels > threshold, one, zero)
    return predicted_labels.eq(labels).sum().item()
