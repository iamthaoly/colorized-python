# NOTE:  This must be the first call in order to work properly!
from deoldify import device
from deoldify.device_id import DeviceId
from os import path
import torch
import os

import fastai
from deoldify.visualize import *
from pathlib import Path
import warnings
import argparse
import ssl

def setupGPU():
    # choices:  CPU, GPU0...GPU7
    device.set(device=DeviceId.GPU0)
    print("-------------------------")
    print("Checking GPU...")
    if torch.backends.mps.is_available():
        print("M1 GPU available.")
    else:
        print("M1 GPU not available.")
        
    if not torch.cuda.is_available():
        print('NVIDIA GPU not available.')
    else:
        print('NVIDIA GPU available')

    print("-------------------------")

def setupTorch():
    torch.backends.cudnn.benchmark=True
    warnings.filterwarnings("ignore", category=UserWarning, message=".*?Your .*? set is empty.*?")

def downloadAndSetupModels():
    print()
    print("Current dir:")
    print(os.system("pwd"))
    os.system("mkdir -p 'models'");

    if not os.path.exists("./models/ColorizeVideo_gen.pth"):
        print("Model file is not existed. Downloading...")
        os.system("curl https://data.deepai.org/deoldify/ColorizeVideo_gen.pth -o ./models/ColorizeVideo_gen.pth")
        # os.system("curl https://media.githubusercontent.com/media/jantic/DeOldify/master/resource_images/watermark.png -o ./resource_images/watermark.png")


def runFirstTime():
    ssl._create_default_https_context = ssl._create_unverified_context
    os.system("pwd")
    setupGPU()
    setupTorch()
    print("---------------------")
    print("Initialing AI...")
    temp = get_video_colorizer()

if __name__ == "__main__":
    runFirstTime()