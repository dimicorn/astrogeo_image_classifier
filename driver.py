from uvfits import UVFits
import matplotlib.pyplot as plt
import os
import yaml
import numpy as np


with open("config.yaml") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

data_path = config['path']
object_dirs = os.listdir(path=data_path)
print(object_dirs[0])
object_uv_data = []
object_map_data = []
for file in os.listdir(data_path + '/' + object_dirs[0]):
    if file[-8:] == 'vis.fits':
        object_uv_data.append(file)
    elif file[-8:] == 'map.fits':
        object_map_data.append(file)

print(object_uv_data)

for data_file in object_uv_data:
    print(data_file)
    data = UVFits(data_path + '/' + object_dirs[0] + '/' + data_file)
    #data.print_info()
    #data.print_uv()
    uu, vv = data.get_uv()
    uu = np.array(uu)
    vv = np.array(vv)
    #print(uu, vv)
    plt.scatter(uu * 1e-6, vv * 1e-6, marker='.')
    plt.xlim(-200, 200)
    plt.ylim(-200, 200)
    plt.xlabel(r'V Baseline projection (M$\lambda$)')
    plt.ylabel(r'U Baseline projection (M$\lambda$)')
    
    plt.title(object_dirs[0], loc='center')
    date = '-'.join(data_file.split('_')[2:5],)
    plt.title(date, loc='left')
    if data_file.split('_')[1] == 'C':
        freq = "4.3 GHz"
    plt.title(freq, loc='right')
    plt.savefig(object_dirs[0] + '.png', dpi=500)
    break
    #plt.show()

# filename = "J0541-1737/J0541-1737_C_2015_12_01_pet_vis.fits"
