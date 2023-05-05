from utils import copy_folder_to_array
import numpy as np

columns=np.array(['ALB','BET', 'DOL', 'LAG', 'OTHER', 'SHARK', 'YFT','missing'])
array = []
sum = 0
for i in columns:
    folder = f'./testingPipeline/cnn/{i}/*.jpg'
    folder_size = copy_folder_to_array(folder,array)
    sum+= folder_size
    print(i,str(folder_size))
print(sum)