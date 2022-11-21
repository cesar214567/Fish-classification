import os, splitfolders
from PIL import Image 
import os 

path = os.path.join('Fish_Dataset', 'Trout')
output = os.path.join('Fish_Dataset', 'Processed')

splitfolders.ratio(path, output=output,
    seed=1337, ratio=(.85, .15), group_prefix=None, move=False) # default values

def change_extension_from_dir(dir):
    for file in os.listdir(dir): 
        if file.endswith(".png"): 
            img = Image.open(os.path.join(dir,file))
            file_name, file_ext = os.path.splitext(file)
            img.save('{}/{}.jpg'.format(dir,file_name))
            os.remove(os.path.join(dir,file))



train_path = os.path.join('.','Fish_Dataset', 'Processed', 'train','Trout')
test_path = os.path.join('.','Fish_Dataset', 'Processed', 'val','Trout')

change_extension_from_dir(train_path)
change_extension_from_dir(test_path)
