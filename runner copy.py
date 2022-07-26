# NOTE:  This must be the first call in order to work properly!
import os
import warnings
from os import path
import torch

import fastai
from deoldify import device
from deoldify.device_id import DeviceId
from deoldify.visualize import *


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
    os.system("mkdir -r 'models'");
    os.system("wget https://data.deepai.org/deoldify/ColorizeVideo_gen.pth - O ./models/ColorizeVideo_gen.pth")
    os.system("wget https://media.githubusercontent.com/media/jantic/DeOldify/master/resource_images/watermark.png - O ./resource_images/watermark.png")

def initColorizer():
    print()

def startColorize(input_paths, render_factor=21):
    colorizer = get_video_colorizer()
    print()
    source_url = '' 
    render_factor = 21  
    watermarked = True 

    for input_path in input_paths:
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
    # downloadAndSetupModels()
    startColorize({"input2.mp4"})

if __name__ == "__main__":
    main()
