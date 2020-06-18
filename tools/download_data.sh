# Download CheXpert small dataset to compute cluster

# Make sure you have access to this data before downloading
# Please read the Stanford University School of Medicine CheXpert
# Dataset Research Use Agreement.
# Information located: https://stanfordmlgroup.github.io/competitions/chexpert/
# Data is approx. 11GB
link="http://download.cs.stanford.edu/deep/CheXpert-v1.0-small.zip"
datadir="/shared/rsaas/pacole2/"

wget $link -P $datadir