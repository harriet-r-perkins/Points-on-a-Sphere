# CENTRAL CONFIGURATIONS - Shell Sort

# Numerical output: number of shells, number of points per shell.
# Graphical outputs: central configs, with spheres coloured according to shell structure.

import numpy as np
import matplotlib.pyplot as plt
import random
from mpl_toolkits.mplot3d import Axes3D

#set number of poins
n=13

x=np.genfromtxt("c-config"+str(n).zfill(3)) #input: needs coordinates of relaxed configuration (for n points).
# for example, for n=13
x=np.array([[ 0.701425, -1.570517,  0.056869],
 [ 1.696018, -0.095868, -0.275839],
 [-0.498407,  0.675145,  1.502506],
 [ 0.498446, -0.675182, -1.502476],
 [-0.701415,  1.570522, -0.056845],
 [-0.954473, -1.292008, -0.617617],
 [ 0.654798,  1.093972, -1.155949],
 [-0.983271,  0.354728, -1.36715 ],
 [-0.654842, -1.093989,  1.155909],
 [ 0.954466,  1.292024,  0.617595],
 [ 0.983269, -0.354744,  1.367148],
 [-0.,       -0.,       -0.      ],
 [-1.696013,  0.095918,  0.275851]])

def norm(x):
    sumsq=0.0
    for k in range(3):
        sumsq=sumsq+(x[k]**2)
    return np.sqrt(sumsq)

rs=np.array([norm(x[i]) for i in range(n)])
r=max(rs)
indexsort=rs.argsort()
rssort=rs[indexsort]
ptsort=x[indexsort]

indexlist=[0]
for i in range(1,n):
    if rssort[i]-rssort[i-1]>r/10.0:
        indexlist.append(i)
indexlist.append(n)

#no. of shells and no of points in shell
numshell=len(indexlist)-1
numpershell=[(indexlist[i+1]-indexlist[i]) for i in range(numshell)]
print ("number of shells = {0:.4f}\n".format(numshell))
print ("number of points in each shell = {0}\n".format(numpershell))


#Render
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
plt.axis('off')
ax.view_init(elev=0.,azim=0)

#group point depending on sphere:
if len(indexlist)>1:
    col=['red','grey','blue','green']        
    for j in range(len(indexlist)-1):#can alter this in order to do individual shells
        x1=[]
        x2=[]
        x3=[]
        for i in range(indexlist[j],indexlist[j+1]):
            x1.append(ptsort[i,0])
            x2.append(ptsort[i,1])
            x3.append(ptsort[i,2])
        ax.scatter(x1,x2,x3,color=col[j],s=8000)
else:
    x1=[]
    x2=[]
    x3=[]
    for i in range(0,n):
        x1.append(x[i,0])
        x2.append(x[i,1])
        x3.append(x[i,2])    
        ax.scatter(x1,x2,x3,color="black",s=500)

ax.set_xlim([-r,r])
ax.set_ylim([-r,r])
ax.set_zlim([-r,r])
ax.set_aspect("equal")


plt.show() 