# CENTRAL CONFIGURATIONS - Radii within config.

# Graphical outputs: graph showing radius of each of the points in a central config.

import numpy as np
import matplotlib.pyplot as plt
import random
from mpl_toolkits.mplot3d import Axes3D

n=38

x=np.genfromtxt("c-config"+str(n).zfill(3)) #input: needs coordinates of relaxed configuration (for n points).
#for example, for n=38
x=np.array([[ 1.511139,  0.068715,  2.217141],
 [ 1.113726, -1.63893,   1.873869],
 [ 0.442152,  2.177461, -1.393195],
 [-1.640494, -1.07346,   1.714521],
 [ 1.640495,  1.073456, -1.714523],
 [ 2.293544, -0.95263,   0.842685],
 [-0.412227, -2.685252, -0.240065],
 [-0.442152, -2.177464,  1.393193],
 [-0.641669,  2.08641,   1.453643],
 [ 1.534194, -0.57165,  -2.126802],
 [ 0.069512,  0.943609,  2.426469],
 [ 0.412228,  2.685252,  0.240061],
 [ 2.336355,  0.706303,  0.908507],
 [ 0.251631,  0.718223, -2.582585],
 [ 1.102671, -2.307197,  0.345974],
 [-2.293547,  0.952635, -0.84268 ],
 [ 0.382292, -0.432237,  1.055708],
 [-1.534191,  0.571646,  2.126803],
 [-2.109078,  1.556935,  0.751901],
 [-2.606886, -0.162596,  0.653507],
 [-0.514941, -1.057524, -0.252891],
 [ 1.225939,  1.807281,  1.574742],
 [-1.102671,  2.307197, -0.345972],
 [ 0.641672, -2.086408, -1.453644],
 [-1.225939, -1.807279, -1.574744],
 [ 1.915235,  1.861611, -0.264226],
 [ 2.606885,  0.162596, -0.653506],
 [-1.113726,  1.638932, -1.873869],
 [-1.01943,   0.364533, 0.524782],
 [ 0.514942,  1.057522,  0.252889],
 [ 1.019427, -0.364536, -0.524785],
 [-0.069514, -0.943608, -2.426471],
 [-1.915236, -1.861611,  0.264221],
 [-0.25163,  -0.718224,  2.582587],
 [ 2.109082, -1.556935, -0.751892],
 [-1.511141, -0.068712, -2.21714 ],
 [-2.336355, -0.706299, -0.908507],
 [-0.382299,  0.432237, -1.055707]])

def norm(x):
    sumsq=0.0
    for k in range(3):
        sumsq=sumsq+(x[k]**2)
    return np.sqrt(sumsq)

rs=np.array([norm(x[i]) for i in range(n)])
indexsort=rs.argsort()
rssort=rs[indexsort]
rlist=[]
for i in range(n):
    rlist.append(rssort[i])

plt.plot(rlist,'r.')
plt.xlabel("Point")
plt.ylabel("Distance from origin")
plt.show()