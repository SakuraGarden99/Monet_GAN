import os
import pathlib

dir_path = "data"
for path in os.listdir(dir_path):
    print(os.path.join(dir_path,path))