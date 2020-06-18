#!/bin/bash

# Not sure what this does?
nvidia-smi

# Activate desired environment
# Need to source bashrc for some reason
source ~/.bashrc
conda activate pytorch-env

# Check to see if data is in the right spot
if [[ ! -d "/data/pacole2" ]]
then
    mkdir /data/pacole2
fi

if [[ ! -d "/data/pacole2/CheXpert-v1.0-small/" ]]
then
    echo "Data is not on gpu storage"
    echo "Copying over data from shared storage"

    FILE="CheXpert-v1.0-small.zip"
    cp /shared/rsaas/pacole2/${FILE} /data/pacole2/

    cd /data/pacole2/
    unzip -qq ${FILE}
    rm ${FILE}
    cd /home/pacole2/
fi

# Data is ready now run python file
cd ~/Projects/xray-view-classifier/
echo "Running python script now"
python main.py --data /data/pacole2/CheXpert-v1.0-small/ --lr 2e-4 --epoch 10 --train
