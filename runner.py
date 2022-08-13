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
    print("Current dir:")
    print(os.system("pwd"))
    os.system("mkdir -p 'models'");

    if not os.path.exists("./models/ColorizeVideo_gen.pth"):
        print("Model file is not existed. Downloading...")
        os.system("curl https://data.deepai.org/deoldify/ColorizeVideo_gen.pth -o ./models/ColorizeVideo_gen.pth")
        os.system("curl https://media.githubusercontent.com/media/jantic/DeOldify/master/resource_images/watermark.png -o ./resource_images/watermark.png")

def initColorizer():
    print()

def startColorize(input_path, render_factor=21):
    colorizer = get_video_colorizer()
    print()
    source_url = '' 
    render_factor = 21  
    watermarked = True 

    if input_path is not None and input_path != '':
        print()
        output_path = colorizer._colorize_from_path(Path(input_path), render_factor)
        print("Video after colorized ->")
        print(output_path)
    # if source_url is not None and source_url !='':
    #     video_path = colorizer.colorize_from_url(source_url, 'video.mp4', render_factor, watermarked=watermarked)
    #     # show_video_in_notebook(video_path)
    # else:
    #     print('Provide a video url and try again.')

def main():
    print("main")
    os.system("pwd")
    setupGPU()
    setupTorch()
    downloadAndSetupModels()
    startColorize("input2.mp4")

def virtualEnv():
    print()
    # sudo apt install python3-venv 
    # which python3.8
    # /usr/bin/python3.10 -m venv venv 
    # source venv/bin/activate 
    # pip install -r requirements.txt


if __name__ == "__main__":
    main()
