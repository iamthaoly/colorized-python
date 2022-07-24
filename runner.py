# NOTE:  This must be the first call in order to work properly!
from deoldify import device
from deoldify.device_id import DeviceId
from os import path
import torch
import os

import fastai
# from deoldify.visualize import *
from pathlib import Path
import warnings

def setupGPU():
    # choices:  CPU, GPU0...GPU7
    device.set(device=DeviceId.GPU0)
    if not torch.cuda.is_available():
        print('GPU not available.')

def setupTorch():
    torch.backends.cudnn.benchmark=True
    warnings.filterwarnings("ignore", category=UserWarning, message=".*?Your .*? set is empty.*?")

def downloadAndSetupModels():
    print()
    os.system("mkdir 'models'");
    os.system("wget https://data.deepai.org/deoldify/ColorizeVideo_gen.pth - O ./models/ColorizeVideo_gen.pth")
    os.system("wget https://media.githubusercontent.com/media/jantic/DeOldify/master/resource_images/watermark.png - O ./resource_images/watermark.png")

# def initColorizer():
#     colorizer = get_video_colorizer()

def main():
    print("main")
    os.system("pwd")
    setupGPU()
    setupTorch()
    # downloadAndSetupModels()


if __name__ == "__main__":
    main()
