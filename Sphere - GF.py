# POINTS ON A SPHERE - Gradient Flow Method

# Numerical outputs: N, final energy, dipole moment. 
# Graphical outputs: Points on sphere. Graph of energy vs iterations. 
# Save to file : coordinates of final configuration.

import numpy as np
import matplotlib.pyplot as plt
import random
from mpl_toolkits.mplot3d import Axes3D

# project one or all points to the sphere
def proj(x,j):
    if(j==n):
        for i in range(0,n):
            norm=np.sqrt(sum(x[i]**2))
            x[i]=x[i]/norm
    else:
        norm=np.sqrt(sum(x[j]**2))
        x[j]=x[j]/norm
    return x
                
# set the number of points
n=10

#set how often to output the energy 
often=50

# set dt/gamma
gamma=0.1
 
# open file to output energy during minimization 
filename="energyout"+str(2)+str(n).zfill(3)
out = open(filename,'w')
out.close()

looplist=[]
energylist=[]

#assign random start points on the sphere
random.seed()
x=proj((2.0*np.random.random((n,3))-1.0),n)
##print x

# calculate the initial energy
energy=0.0
for i in range(0,n):
    for j in range(i+1,n):
        distance=np.sqrt(sum((x[i]-x[j])**2))
        energy=energy+1.0/distance 


#function to calculate norm of a vector
def norm(x):
    sumsq=0.0
    for k in range(3):
        sumsq=sumsq+(x[k]**2)
    return np.sqrt(sumsq)
    

#the main loop to reduce the energy       
iteration=0.0

#for loop in range(maxloop):
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
                
        #calculate radial component
        dotp=0.0
        for a in range(3):
            dotp=dotp+F[a]*x[i,a]
        radcomp=dotp*x[i]
        
        #calculate force acting along suface of sphere (Ftil=F-radcomp)
        Ftil=F-radcomp
            
        #move position of point i in direction of force
        x[i]=x[i]+gamma*Ftil
        x=proj(x,i)
        
        # calculate the difference in energy
        difference=0.0
        for j in range(n):
            if(j!=i):
                distance=np.sqrt(sum((x[i]-x[j])**2))
                distanceold=np.sqrt(sum((old-x[j])**2))
                difference=difference+1.0/distance-1.0/distanceold;
        
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
    
    if(iteration%often==0):
        looplist.append(iteration)
        energylist.append(energy)

# output number of iterations and final energy to the screen
print("N = {0}\n".format(n))
print("Number of iterations = {0:.1f} \n".format(iteration))
print("Final energy = {0:.6f} \n".format(energy))

# output points to a file        
filename2="pointsout"+str(n).zfill(3)
points=open(filename2,'w')
for i in range(0,n):
    for j in range(0,3):
        points.write("{0:.6f} ".format(x[i,j]))
    points.write('\n')              
points.close()

##calculate dipole moment for each n
dpm=np.sum(x,axis=0)
D=norm(dpm)
print("Dipole moment = {0:.4f}\n".format(D))

#Create a sphere
theta, phi = np.mgrid[0.0:np.pi:100j, 0.0:2.0*np.pi:100j]
xs = np.sin(theta)*np.cos(phi)
ys = np.sin(theta)*np.sin(phi)
zs = np.cos(theta)

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
ax = fig.add_subplot(211, projection='3d')
ax.plot_surface(xs, ys, zs,  rstride=4, cstride=4, color='yellow', alpha=0.3, linewidth=0)
ax.scatter(x1,x2,x3,color='black',s=80)
plt.axis('off')
ax.view_init(elev=0.,azim=0)

ax.set_xlim([-1.0,1.0])
ax.set_ylim([-1.0,1.0])
ax.set_zlim([-1.0,1.0])
ax.set_aspect("equal")
ax.set_title("{0} ".format(n)+"points on a sphere")

ax2=fig.add_subplot(212)
ax2.plot(looplist,energylist)
ax2.set_xlabel("iteration number")
ax2.set_ylabel("energy")

plt.show() 