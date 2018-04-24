# POINTS ON A DISK - Voronoi Regions

# Graphical outputs: voronoi construction for points on a disk

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
x=np.genfromtxt("disk"+str(n).zfill(3)) #input: needs coordinates of relaxed configuration (for n points).

#for example, for n=10 points
x=np.array([[-0.933063, -0.359712],
 [ 0.933063,  0.359712],
 [-0.630439,  0.776239],
 [ 0.543431,  0.839454],
 [-0.543431, -0.839454],
 [-0.966298,  0.257428],
 [ 0.053774, -0.998553],
 [-0.053774,  0.998553],
 [ 0.966298, -0.257428],
 [ 0.630439, -0.776239]])

#create m**2 points on disk
m=200
theta, r = np.mgrid[0.0:2*np.pi:m*1j, 0.0:1.0:m*1j]
xs = r*np.cos(theta)
ys = r*np.sin(theta)


#create colour matrix
colmat=np.zeros((m,m))
for m1 in range(m):
    for m2 in range(m):
        p=[xs[m1,m2],ys[m1,m2]]
        dmin=4.0
        for i in range(0,n):
            d=dist(x[i],p)
            if d<dmin:
                dmin=d
                nrstpnt=i
        colmat[m1,m2]=nrstpnt/float(n)
        
        
x1=[]
x2=[]
col=[]
for i in range(m):
    for j in range(m):
        x1.append(xs[i,j])
        x2.append(ys[i,j])
        col.append(colmat[i,j])

x11=np.array(x1)
x22=np.array(x2)
coll=np.array(col)

    
#Render
fig = plt.figure()
ax = fig.add_subplot(111, aspect='equal')
ax.scatter(x11,x22,s=40,c=coll)
plt.axis('off')

ax.set_xlim([-1.5,1.5])
ax.set_ylim([-1.5,1.5])
ax.set_aspect("equal")
ax.set_title("{0} ".format(n)+"disk voronoi regions")
          
plt.show()