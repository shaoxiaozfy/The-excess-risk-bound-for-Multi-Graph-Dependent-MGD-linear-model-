import os,sys
root_path = os.path.abspath(__file__)
root_path = '/'.join(root_path.split('/')[:-2])
print(root_path)
sys.path.append(root_path)

from .image_data_scripts import *