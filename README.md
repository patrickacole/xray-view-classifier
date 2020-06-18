# Classifier for predicting the orientation of radiographs
-------------------------------

## Training the model
This code will train a classifier to predict the orientation of a radiograph

usage: main.py [-h] [--data DATA] [--lr LR] [--epochs EPOCHS] [--batch BATCH]
               [--checkpointdir CHECKPOINTDIR] [--train] [--test] [--plot]
               [--prefix PREFIX] [--load]

Arguments for training

optional arguments: <br/>
  -h, --help            show this help message and exit <br/>
  --data DATA           Path to where CheXpert data is stored <br/>
  --lr LR               Learning rate <br/>
  --epochs EPOCHS       Number of epochs to train <br/>
  --batch BATCH         Batch size to use while training <br/>
  --checkpointdir CHECKPOINTDIR
                        Path to checkpoint directory <br/>
  --train               Flag to train <br/>
  --test                Flag to evaluate on test data <br/>
  --plot                Flag to generate plot of score <br/>
  --prefix PREFIX       prefix for model to load ("last" or "best") <br/>
  --load                Flag to load a model <br/>