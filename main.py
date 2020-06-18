import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt

from time import time
from argparse import ArgumentParser
from torch.utils.data import DataLoader

# custom imports
from modules.resnet18 import ResNet18
from utils.dataset import *
from utils.metrics import *
from utils.checkpoints import *

# global variables
device = None
args = None

def args_parse():
    """
    Returns command line parsed arguments
    @return args: commandline arguments (Namespace)
    """
    parser = ArgumentParser(description="Arguments for training")
    parser.add_argument('--data', default="~/data/", help="Path to where CheXpert data is stored")
    parser.add_argument('--lr', default=2e-4, type=float, help="Learning rate")
    parser.add_argument('--epochs', default=100, type=int, help="Number of epochs to train")
    parser.add_argument('--batch', default=128, type=int, help="Batch size to use while training")
    parser.add_argument('--checkpointdir', default="checkpoints/", help="Path to checkpoint directory")
    parser.add_argument('--train', default=False, action="store_true", help="Flag to train")
    parser.add_argument('--test', default=False, action="store_true", help="Flag to evaluate on test data")
    parser.add_argument('--plot', default=False, action="store_true", help="Flag to generate plot of score")
    parser.add_argument('--prefix', default="best", help="prefix for model to load (\"last\" or \"best\"")
    parser.add_argument('--load', default=False, action="store_true", help="Flag to load a model")
    return parser.parse_args()

def plot_curve(train, test, curvetype, directory="outputs/"):
    """
    Plots the given train and test curves
    @param train    : train values (NumPy array)
    @param test     : test values (NumPy array)
    @param curvetype: "Accuracy" or "Loss" (string)
    @param directory: directory to store plot (string)
    """
    if not os.path.exists(directory):
        os.makedirs(directory)

    plt.figure('{} vs Epoch'.format(curvetype))
    plt.plot(np.arange(len(train)), train, '-C0', label="Training")
    plt.plot(np.arange(len(test)), test, '-C1', label="Testing")
    plt.title('Training and Testing {} vs Epoch'.format(curvetype))
    plt.xlabel('Epoch')
    plt.ylabel('{}'.format(curvetype))
    plt.legend()
    plt.savefig(os.path.join(directory,curvetype.lower() + '.png'),
                dpi=500.0, bbox_inches='tight')

def save_curves(train, test, curvetype, directory="outputs/"):
    if not os.path.exists(directory):
        os.makedirs(directory)

    filename = os.path.join(directory, curvetype.lower() + '.npz')
    np.savez(filename, train=train, test=test)

def train_epoch(epoch, model, trainloader, optimizer, criterion):
    """
    Train the model for one epoch
    @param  model         : the model to be trained (nn.Module)
    @param  trainloader   : the data loader with the training data (DataLoader)
    @param  optimizer     : optimizer to use for training (torch.optim)
    @param  criterion     : loss function to evaluate with (nn.Loss)
    @return train_loss    : average loss over training one epoch (float)
    @return train_score   : average score over one epoch (float)
    """
    model.train()
    train_loss = 0.0
    train_score = 0.0
    total_predicts = 0
    for i, (images, labels) in enumerate(trainloader):
        # Set images and labels to be on the current device
        images = images.to(device)
        labels = labels.to(device)
        # Repeat along first dimension because model expects 3 channels
        images = images.repeat(1, 3, 1, 1)
        # Zero out gradients
        optimizer.zero_grad()
        # Get model predictions
        predicts = model(images)
        # Calculate Cross Entropy Loss
        loss = criterion(predicts, labels)
        # Compute gradients with backward pass
        loss.backward()
        # Update weights
        optimizer.step()
        # Update running averages of loss and score
        train_loss += loss.item()
        total_predicts += images.size(0)
        train_score += accuracy_calculation(predicts, labels)
        del images, labels, predicts

    return (train_loss / (i + 1), 100 * (train_score / total_predicts))

def test_epoch(model, testloader, criterion):
    """
    Test the model for one epoch
    @param  model         : the model to be trained (nn.Module)
    @param  trainloader   : the data loader with the training data (DataLoader)
    @param  criterion     : loss function to evaluate with (nn.Loss)
    @return test_loss     : test loss for current epoch (float)
    @return test_score    : test score for current epoch (float)
    """
    model.eval()
    test_loss = 0.0
    test_score = 0.0
    total_predicts = 0
    for i, (images, labels) in enumerate(testloader):
        # Set images and labels to be on the current device
        images = images.to(device)
        labels = labels.to(device)
        # Repeat along first dimension because model expects 3 channels
        images = images.repeat(1, 3, 1, 1)
        # Get model predictions
        predicts = model(images)
        # Calculate Cross Entropy Loss
        loss = criterion(predicts, labels)
        # Update running averages of loss and score
        test_loss += loss.item()
        total_predicts += images.size(0)
        test_score += accuracy_calculation(predicts, labels)
        del images, labels, predicts

    return (test_loss / (i + 1), 100 * (test_score / total_predicts))

def train(model, trainloader, testloader):
    """
    Train the model
    @param  model         : the model to be trained (nn.Module)
    @param  trainloader   : the data loader with the training data (DataLoader)
    @param  testloader    : the data loader with the testing data (DataLoader)
    """
    # Create Adam optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.lr,
                           betas=(0.9, 0.999))
    # Loss function is cross entropy
    criterion = nn.BCEWithLogitsLoss()
    # Learning Rate Scheduler - decay when curve plateaus
    lrscheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.2)

    # set up variables to store each epochs score and loss
    best_score = 0.0
    best_loss = np.inf
    train_scores = np.array([0.0], dtype=np.float32)
    train_losses = np.array([], dtype=np.float32)
    test_scores = np.array([0.0], dtype=np.float32)
    test_losses = np.array([], dtype=np.float32)

    for e in range(args.epochs):
        # train over training data
        train_time = time()
        train_loss, train_score = train_epoch(e, model, trainloader, optimizer, criterion)
        train_time = time() - train_time

        # store average of training score and loss
        train_scores = np.append(train_scores, train_score)
        train_losses = np.append(train_losses, train_loss)
        print("Epoch [{} / {}]: Training loss: {:0.4f}, Training score: {:0.2f}" \
              .format(e + 1, args.epochs, train_loss, train_score), end = ', ')

        # evaluate current model on test set
        test_time = time()
        test_loss, test_score = test_epoch(model, testloader, criterion)
        test_time = time() - test_time

        print("Test loss: {:0.4f}, Test score: {:0.2f}" \
              .format(test_loss, test_score), end = ', ')

        print("Epoch elapsed time: [Training: {:0.2f}, Test: {:0.2f}] secs" \
              .format(train_time, test_time), end='\n')

        # store average of test score and loss
        test_scores = np.append(test_scores, test_score)
        test_losses = np.append(test_losses, test_loss)

        # check to see if current model parameters perform the best
        isbest = test_score > best_score
        if isbest:
            best_score = test_score
            best_loss = test_loss

        # save current model
        # training_state = {'epoch'      : e + 1,
        #                   'state_dict' : model.state_dict(),
        #                   'optim_dict' : optimizer.state_dict()}
        training_state = {'state_dict' : model.state_dict()}
        save_checkpoint(training_state, isbest=isbest,
                        checkpoint=args.checkpointdir)

        # lrscheduler step
        lrscheduler.step()

        # save curves
        save_curves(train_scores, test_scores, "Accuracy")
        save_curves(train_losses, test_losses, "Loss")

def test(model, testloader):
    """
    Evaluate model on testing data
    @param  model        : the model to be trained (nn.Module)
    @param  testloader   : the data loader with the training data (DataLoader)
    @return test_loss    : average test loss (float)
    @return test_score   : average test score (float)
    """
    # Loss function is cross entropy
    criterion = nn.BCEWithLogitsLoss()
    # Set model to eval mode
    model.eval()

    with torch.no_grad():
        avg_test_loss, avg_test_acc = test_epoch(model, testloader, criterion)

    print("Pretrained ResNet34 trained on CheXpert produced an accuracy score of {:0.4f} with a loss value of {:0.4f}" \
          .format(avg_test_acc, avg_test_loss))


if __name__=="__main__":
    args = args_parse()
    assert args.train or args.test or args.plot, "You must either use the train, test, and/or plot flag, use \'-h\' to see usage."
    assert not (args.train and args.test), "You should only choose the flag to train or to test, not both."

    # Print arguments used for training
    print("Using the following hyperparemters:")
    print("Data:                 " + args.data)
    print("Learning rate:        " + str(args.lr))
    print("Number of epochs:     " + str(args.epochs))
    print("Batch size:           " + str(args.batch))
    print("Checkpoint directory: " + args.checkpointdir)
    print("Train:                " + str(args.train))
    print("Test:                 " + str(args.test))
    print("Plot:                 " + str(args.plot))
    print("Prefix:               " + args.prefix)
    print("Load:                 " + str(args.load))
    print("Cuda:                 " + str(torch.cuda.device_count()))
    print("")

    device = torch.device(("cpu","cuda:0")[torch.cuda.is_available()])
    model = ResNet18(1, pretrained=True).to(device)

    if args.load or args.test:
        load_checkpoint(args.checkpointdir, args.prefix, model)

    if (torch.cuda.device_count() > 1):
        device_ids = list(range(torch.cuda.device_count()))
        print("GPU devices being used: ", device_ids)
        model = nn.DataParallel(model, device_ids=device_ids)

    # Set up transforms for training and test
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.502845,), (0.290049,))])

    # For trainning
    if args.train:
        trainset = CheXpertDataset(args.data, img_size=256, train=True, transforms=transform)
        testset = CheXpertDataset(args.data, img_size=256, train=False, transforms=transform)

        trainloader = torch.utils.data.DataLoader(trainset,
                batch_size=args.batch, shuffle=True, num_workers=8)
        testloader = torch.utils.data.DataLoader(testset,
                batch_size=args.batch, shuffle=False, num_workers=8)

        train(model, trainloader, testloader)

    # For testing
    if args.test:
        testset = CheXpertDataset(args.data, img_size=256, train=False, transforms=transform)
        testloader = torch.utils.data.DataLoader(testset,
                batch_size=args.batch, shuffle=False, num_workers=8)

        test(model, testloader)

    # Plot score and loss curves
    if args.plot:
        # score plot
        fptr = np.load("outputs/accuracy.npz")
        plot_curve(fptr["train"], fptr["test"], "Accuracy")

        # loss plot
        fptr = np.load("outputs/loss.npz")
        plot_curve(fptr["train"], fptr["test"], "Loss")
