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
        # os.system("curl https://media.githubusercontent.com/media/jantic/DeOldify/master/resource_images/watermark.png -o ./resource_images/watermark.png")

def initColorizer():
    print()

def moveVideoToPath(init_path, output_path):
    os.system("mv %s %s" % (init_path, output_path))
    print("Done move %s to %s" % (init_path, output_path))

def startColorize(input_paths, output_paths, render_factor=21):
    colorizer = get_video_colorizer()
    print()
    # render_factor = 21  
    for i in range(0, len(input_paths)):
        input_path = input_paths[i]
        output_path = output_paths[i]
        if input_path is not None and input_path != '':
            print()
            video_path = colorizer._colorize_from_path(Path(input_path), render_factor)
            print("Video path after colorized ->")
            print(video_path)
            moveVideoToPath(video_path, output_path)

def main():
    print("main")
    os.system("pwd")
    setupGPU()
    setupTorch()
    downloadAndSetupModels()
    startColorize("input3.mp4")

def testParser():
    DELIMETER = ";"
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output", help = "Array of input path")
    parser.add_argument("-r", "--render", help = "Render factor")
    parser.add_argument("-i", "--input", help = "Array of output path")

    # Read arguments from command line
    args = parser.parse_args()
    
    if args.output and args.input and args.render:
        print("Raw input: % s" % args.input)
        print("Raw output: % s" % args.output)
        print("Render number: % s" % args.render)

        inputs = str(args.input).split(DELIMETER)
        outputs = str(args.output).split(DELIMETER)
        render_factor = int(args.render)

        print(inputs)
        print(outputs)
        print(render_factor)
        
        if (len(inputs) != len(outputs)):
            print("ERROR: Input length != Output length.")
            return
        startColorize(inputs, outputs, render_factor)
    else:
        print("ERROR: Some arguments are empty. Please check again.")

    
if __name__ == "__main__":
    # main()
    testParser()
    # Arguments: [Input], [Output], Render factor.
    # input: /mnt/D/Code/DeOldify/input3.mp4
    # output: /mnt/D/Code/DeOldify/input3_color.mp4