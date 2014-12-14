
from lbfgs import *


def fml (pos):
    return (pos[0]-2.0)**2+(pos[1]-4.0)**2;

def deltaFml (pos):
    return np.array([2*(pos[0]+20.0), 2*(pos[1]-43.0)])

lb = LBFGS()
lb.initialize ()


positions = np.array([5.0, 8.0])
for i in xrange (5000):
    forces = deltaFml (positions)
    
    positions = lb.step (positions, forces)
    print "i= ",i,"   positions: ",positions,"    forces= ",forces 

    if np.max(forces) < 0.0001:
        break

