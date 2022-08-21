from fastai.core import *
from fastai.vision import *
import time

for i in progress_bar(range(0, 5)):
    time.sleep(1)
    print(i)
