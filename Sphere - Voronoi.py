# POINTS ON A SPHERE - Voronoi Regions

# Graphical outputs: voronoi construction for points on a sphere

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def dist(x1,x2):
    sum=0.0
    for i in range(len(x1)):
        sum=sum+(x1[i]-x2[i])**2
    return np.sqrt(sum)

#choose n
n=10

#call points
x=np.genfromtxt("pointsout"+str(n).zfill(3)) #input: needs coordinates of relaxed configuration (for n points).
#for example, for n=10
x=np.array([[ 0.618473, -0.310726,  0.721762],
 [-0.401059,  0.034002,  0.915421],
 [ 0.506709, -0.836481, -0.208677],
 [ 0.544458,  0.73373,   0.406455],
 [-0.0273,    0.673138, -0.739013],
 [-0.454401, -0.848354,  0.27169 ],
 [-0.506708,  0.836482,  0.208674],
 [-0.190115, -0.396415, -0.898171],
 [-0.972817, -0.026591, -0.230044],
 [ 0.882761,  0.141214, -0.448098]])

#Create a m**2 points on sphere
m=200
theta, phi = np.mgrid[0.0:np.pi:m*1j, 0.0:2.0*np.pi:m*1j]
xs = np.sin(theta)*np.cos(phi)
ys = np.sin(theta)*np.sin(phi)
zs = np.cos(theta)

#create colour matrix
colmat=np.zeros((m,m))
for m1 in range(m):
    for m2 in range(m):
        p=[xs[m1,m2],ys[m1,m2],zs[m1,m2]]
        dmin=4.0
        for i in range(0,n):
            d=dist(x[i],p)
            if d<dmin:
                dmin=d
                nrstpnt=i
        colmat[m1,m2]=nrstpnt/float(n)

x1=[]
x2=[]
x3=[]
col=[]
for i in range(m):
    for j in range(m):
        x1.append(xs[i,j])
        x2.append(ys[i,j])
        x3.append(zs[i,j])
        col.append(colmat[i,j])

x11=np.array(x1)
x22=np.array(x2)
x33=np.array(x3)
coll=np.array(col)

    
#Render
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x11,x22,x33,s=40,c=coll)
plt.axis('off')
ax.view_init(azim=90.0,elev=90.0)

ax.set_xlim([-1.0,1.0])
ax.set_ylim([-1.0,1.0])
ax.set_zlim([-1.0,1.0])
ax.set_aspect("equal")
ax.set_title("{0} ".format(n)+"voronoi regions")
          
plt.show()