# read result from file json
# startColorize(input, output, render_factor)
import json
from runner import startColorize

input = ""
output = ""
render = ""
# TODO: Change json_path
json_path = "userVideo.txt"

print("Start reading json file...")
with open(json_path) as f:
    data = json.load(f)
    print(data["input_video"])
    print(data["output_video"])
    print(data["render_factor"])
    input = data["input_video"]
    output = data["output_video"]
    render = data["render_factor"]

startColorize(input_paths=[input], output_paths=[output], render_factor=render)