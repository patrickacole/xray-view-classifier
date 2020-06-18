import torch
import torch.nn as nn
import torchvision.models as models

from torchvision.models.resnet import model_urls

class ResNet18(nn.Module):
    # Train a pretrained ResNet18
    def __init__(self, num_classes, pretrained=True, last_layer_only=False):
        """
        Init function

        @param num_classes : the number of outputs for the model
        @param last_layer_only : whether to only fine tune the last layer of the model
        """
        super(ResNet18, self).__init__()

        # Load pre-trained ResNet Model
        model_urls['resnet18'] = model_urls['resnet18'].replace('https://', 'http://')
        self.resnet18 = models.resnet18(pretrained=pretrained)

        # Set gradients to false
        if last_layer_only:
            for param in self.resnet18.parameters():
                param.requires_grad = False

        # Replace last fc layer
        num_feats = self.resnet18.fc.in_features

        # Replace fc layer in resnet to a linear layer of size (num_feats, num_classes)
        self.resnet18.fc = nn.Linear(num_feats, num_classes)

    def forward(self, x):
        """
        Forward pass

        @param x : input batch images of shape(B, 1, H, W)
        @return predicts : score predictions for each class
        """
        return self.resnet18(x)


if __name__=="__main__":
    x = torch.zeros(4, 3, 128, 128)
    model = ResNet18(1)
    print(model(x).shape)