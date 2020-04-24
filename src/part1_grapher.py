from os import listdir

import matplotlib.pyplot as plt
import numpy as np

case = 'GEO'
file_names = [f for f in listdir('Data/'+case)]

data = []
for files in file_names:
    exec("data.append(np.squeeze(np.load('Data/%s/'+files)['x']))" % (case))

pos_data_dic = {}
i=0
key_names = [f.strip('.npz') for f in file_names]
for key in key_names:
    pos_data_dic[key] = data[i]
    i=i+1

true = pos_data_dic['true_'+case]
subset = [ 'two_body_diff', 'drag_diff', 'J2_diff', 'J23_diff', 'SRP_diff', 'three_body_diff']
i=0
time = np.arange(0, 5, 10/(60*24))
for item in subset:
    exec("%s = np.abs(true - pos_data_dic[key_names[i]])" % (item))
    exec("plt.plot(time, %s)" % item)
    plt.yscale('log')
    plt.ylabel('Difference (km)')
    plt.xlabel("Time, Days from Epoch")
    plt.title(case+" Force Model Comparisons")
    plt.legend(["Sun", "Drag", "J2", "J2 & J3", "SRP", "Sun & Moon"])
    i = i+1
plt.show()
