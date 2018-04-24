# CENTRAL CONFIGURATIONS - Gradient Flow

# Numerical outputs: N, energy, radius of config.
# Graphical output: Points in ball.

import numpy as np
import matplotlib.pyplot as plt
import random
from mpl_toolkits.mplot3d import Axes3D
                

def norm(x):
    sumsq=0.0
    for k in range(3):
        sumsq=sumsq+(x[k]**2)
    return np.sqrt(sumsq)

#set the number of points    
n=13 

#assign random start points
random.seed()
x=2.0*np.random.random((n,3))-1.0

# calculate the initial energy
energy=0.0
for i in range(0,n):
    energyp1=0.0
    for j in range(i+1,n):
        energyp1=energyp1+1.0/norm((x[i]-x[j]))
    energyp2=0.5*norm(x[i])**2
    energy=energy+energyp1+energyp2    

#the main loop to reduce the energy       
iteration=0.0

gamma=0.05

while True:

    for i in range(n):
        
        # store the old coordinates of point i
        old=np.array(x[i])

        #calculate force acting on each point i
        F=np.zeros(3)
        for j in range(n):
            if j!=i:
                dist=x[i]-x[j]
                normdist=norm(dist)
                F=F+(dist/(normdist**3))
        F=F-x[i]
            
        #move position of point i in direction of force
        x[i]=x[i]+gamma*F

        # calculate the difference in energy
        difference=0.0
        for j in range(n):
            if(j!=i):
                distance=norm((x[i]-x[j]))
                distanceold=norm((old-x[j]))
                difference=difference+1.0/distance-1.0/distanceold
        difference=difference+0.5*(norm(x[i])**2-norm(old)**2)
        
        #sum energy
        if(difference<0.0):
            energy=energy+difference
        else:
            x[i]=old
            gamma=gamma/2
        
    if gamma<0.00001:
        break
    else:
        iteration=iteration+1

r=max(norm(x[i]) for i in range(n))

# return number of iterations final energy and radius of configuration
print("{0} {1:.5f} {2:.5f}\n".format(n,energy,r))


#Create a sphere
theta, phi = np.mgrid[0.0:np.pi:100j, 0.0:2.0*np.pi:100j]
xs = r*np.sin(theta)*np.cos(phi)
ys = r*np.sin(theta)*np.sin(phi)
zs = r*np.cos(theta)

#convert data
x1=[]
x2=[]
x3=[]
for i in range(0,n):
    x1.append(x[i,0])
    x2.append(x[i,1])
    x3.append(x[i,2])
    

#Render
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(xs, ys, zs,  rstride=4, cstride=4, color='yellow', alpha=0.3, linewidth=0)
ax.scatter(x1,x2,x3,color='black',s=200)
plt.axis('off')
ax.view_init(elev=0.,azim=0)

ax.set_xlim([-r,r])
ax.set_ylim([-r,r])
ax.set_zlim([-r,r])
ax.set_aspect("equal")
ax.set_title("{0} ".format(n)+"central config.")

plt.show() 