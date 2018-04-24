# POINTS ON A DISK - Rings

# Graphical output: ring structure .

import matplotlib.pyplot as plt
import numpy as np

n=10
    
#call points
X=np.genfromtxt("disk"+str(n).zfill(3)) #input: needs coordinates of relaxed configuration (for n points).

#for example, for n=10 points
X=np.array([[-0.933063, -0.359712],
 [ 0.933063,  0.359712],
 [-0.630439,  0.776239],
 [ 0.543431,  0.839454],
 [-0.543431, -0.839454],
 [-0.966298,  0.257428],
 [ 0.053774, -0.998553],
 [-0.053774,  0.998553],
 [ 0.966298, -0.257428],
 [ 0.630439, -0.776239]])


def norm(x):
    sumsq=0.0
    for k in range(2):
        sumsq=sumsq+(x[k]**2)
    return np.sqrt(sumsq)

#work out distance of each point from centre
rs=[norm(X[i]) for i in range(n)]
rs=np.array(rs)

#work out which (if any) of these distance differ
inds=rs.argsort()
sortrs=rs[inds]

r=[sortrs[0]]
for i in range(n-1):
    if sortrs[i+1]-sortrs[i]>0.1:
        r.append(sortrs[i+1])

#Render
fig = plt.figure()
ax = fig.add_subplot(111,aspect='equal')
ax.set_xlim([-1.5,1.5])
ax.set_ylim([-1.5,1.5])
ax.set_aspect("equal")
ax.set_title("{0} ".format(n)+"points on a disk")
plt.axis('off')

#Create mulitple rings with radii from r
if r[0]<0.01:
    ax.scatter(0,0,color='black',s=20)
    
for i in range(len(r)):
    theta= np.linspace(0,np.pi*2,100)
    xs = r[i]*np.cos(theta)
    ys = r[i]*np.sin(theta)
    ax.plot(xs, ys, color='black', linewidth=2)
    plt.axis('off')

plt.show() 