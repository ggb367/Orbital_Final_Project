from os import listdir

import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl

cases = ['Molyniya', 'GEO', 'LEO', 'MEO']
from cycler import cycler
mpl.rcParams['axes.prop_cycle'] = cycler(color='bgrcmyk')
for case in cases:
    file_names = [f for f in listdir('Data/'+case)]

    data = []
    for files in file_names:
        exec("data.append(np.squeeze(np.load('Data/%s/'+files)['x']))" % (case))

    pos_data_dic = {}
    i=0
    key_names = [f.strip('.npz') for f in file_names]
    for key in key_names:
        pos_data_dic[key] = (data[i])
        i=i+1

    true = pos_data_dic['true_'+case]
    subset = ['two_body_diff', 'drag_diff', 'J2_diff', 'J23_diff', 'SRP_diff', 'three_body_diff']
    i=0
    time = np.arange(0, 5, 10/(60*24))
    ls = [(0, (5, 10)), '--', 'solid', (0, (1, 1)),  (0, (3, 10, 1, 10)), (0, (1, 10))]
    lables = ["Two Body", "Drag", "J2", "J2 & J3", "SRP", "Sun & Moon"]
    plt.figure()
    for item in subset:
        exec("%s = (pos_data_dic[key_names[i]]-true)" % (item))
        temp = []
        exec("for row in range(np.shape(%s)[0]): temp.append(np.linalg.norm(%s[row,:]))"% (item, item))
        exec("%s = temp" % (item))
        exec("plt.plot(time, (%s), label = lables[i], linestyle = ls[i])" % item)
        i = i+1

    plt.yscale('log')
    plt.ylabel('Error (km)')
    plt.xlabel("Time, Days from Epoch")
    if case is 'Molyniya':
        plt.title("Molniya Force Model Comparisons")
    else:
        plt.title(case+" Force Model Comparisons")
    plt.legend()
    plt.xlim((0, 5))
    # plt.axis('equal')
plt.show()
