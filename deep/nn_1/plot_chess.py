
import matplotlib.pyplot as plt
from numpy.random import rand
import numpy as np
import sys
data = np.genfromtxt('chess_data.csv', delimiter=',', skip_header=1,
                     skip_footer=0, names=['id', 'x', 'y', 'weight', 'label'], dtype=None)

#print data['label']
#sys.exit ()

lab = []
for l in data['label']:
    if l == 's':
        lab.append ('red')
    else:
        lab.append ('blue')

#print lab
data['label'] = lab

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.scatter(data['x'], data['y'], color=data['label'], s = 0.1, label='test', alpha = 0.8) #, edgecolors='none')


plt.show()
