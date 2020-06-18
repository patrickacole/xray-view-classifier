import os
import numpy as np
import torch
import torchvision.transforms as transforms

from argparse import ArgumentParser

# custom imports
from modules.resnet18 import ResNet18
from utils.checkpoints import *


def args_parse():
    """
    Returns command line parsed arguments
    @return args: commandline arguments (Namespace)
    """
    parser = ArgumentParser(description="Arguments for training")
    parser.add_argument('datapath', help="Path to where data is stored. See README.md for desired directory structure.")
    parser.add_argument('--output_dir', default=None, help="If given where the data will copied to be stored.")
    parser.add_argument('--model_path', default='./checkpoints/last.pth', help="Path to file which stores the model state dict.")
    return parser.parse_args()

def sort_directory():
    # parse commandline arguments
    args = args_parse()

    # Print arguments for sorting
    print("Using the following parameters:")
    print("Data path:            " + args.datapath)
    print("Output directory:     " + str(args.output_dir))
    print("Model path:           " + args.model_path)
    print("")

    # use cpu since only one image will be passed through at once
    device = torch.device("cpu")

    # initialize model
    model = ResNet18(1, pretrained=False).to(device)

    # get model prefix and directory name
    prefix = os.path.splitext(os.path.basename(args.model_path))[0]
    checkpointdir = os.path.dirname(args.model_path)
    load_checkpoint(checkpointdir, prefix, model)

    # get desired transform needed for the model
    transform = transforms.Compose([transforms.Resize((256, 256)), # model was trained with images of size 256x256
                                    transforms.ToTensor(),         # transform PIL image to tensor
                                    transforms.Normalize((0.502845,), (0.290049,))]) # mean and std of original dataset

    # check if output directory given
    if args.output_dir == None:
        # remove the last directory from the data path
        # os sep is '/' on unix based and '\' on windows based
        basedir = os.sep.join(args.datapath.split(os.sep)[:-1])
        # add new directory name
        args.output_dir = os.path.join(basedir, 'sorted-data')

    # check if output directory exists
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # parse through all directories in directory
    # expected data structure:
    # data
    #   |
    #   |-- ACC_NUM
    #           |
    #           |-- AMSE0001
    #           |       |
    #           |       |-- radiographs
    #           |-- AMSE0002
    #           |       |
    #           |       |-- radiographs
    #           |-- KeyImages
    #
    # Contents of key images will be ignored

    for acc_num in os.listdir(args.datapath):
        # get the outer path for the acc_num
        outer_path = os.path.join(args.datapath, acc_num)
        # check if acc_num is a directory; if not skip
        if not os.path.isdir(outer_path):
            continue

        # keep track of the current frontal and lateral radiograph number for each acc_num
        frontal_num = 1
        lateral_num = 1

        # parse through the inner folders
        for image_folder in os.listdir(outer_path):
            # inner path for the acc_num and current folder
            inner_path = os.path.join(outer_path, image_folder)
            # if image_folder is KeyImages or image_folder isn't a directory, skip
            if image_folder == 'KeyImages' or not os.path.isdir(inner_path):
                continue

            # parse through all the images inside the inner path
            for image_file in os.listdir(inner_path):
                # make sure the file is a png
                if '.png' not in image_file:
                    continue

                # get image path and open image in grayscale
                image_path = os.path.join(inner_path, image_file)
                image = Image.open(image_path).convert('L')
                # get image tensor, shape (1x1xHxW)
                image_tensor = transform(image).unsqueeze(0)
                # image tensor needs 3 channels to be passed into the model
                image_tensor = image_tensor.repeat(1, 3, 1, 1)
                # get the output score of the model
                score = model(image_tensor)
                # determine if the image is a frontal or lateral
                radiograph_type = None
                if score[0] > 0.5:
                    radiograph_type = 'Frontal-{}'.format(frontal_num)
                    frontal_num += 1
                else:
                    radiograph_type = 'Lateral-{}'.format(lateral_num)
                    lateral_num += 1

                # save image in correct space
                save_file = "{}-{}.png".format(acc_num, radiograph_type)
                save_path = os.path.join(args.output_dir, save_file)
                image.save(save_path)

                # free up some space
                del image, image_tensor, score


if __name__=="__main__":
    sort_directory()