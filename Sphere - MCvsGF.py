#POINTS ON A SPHERE - Monte Carlo vs Gradient Flow

#Graphical Output: Initial config, Final config. Graph of energy vs time for monte carlo and gradient flow methods.

import numpy as np
import matplotlib.pyplot as plt
import random
from mpl_toolkits.mplot3d import Axes3D
import time

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
n=6

#assign random start points on the sphere
random.seed()
X=proj((2.0*np.random.random((n,3))-1.0),n)    
filename="random"+str(n).zfill(3)
points=open(filename,'w')
for i in range(0,n):
    for j in range(0,3):
        points.write("{0:.6f} ".format(X[i,j]))
    points.write('\n')              
points.close()

#function to calculate norm of a vector
def norm(x):
    sumsq=0.0
    for k in range(3):
        sumsq=sumsq+(x[k]**2)
    return np.sqrt(sumsq)
    
#monte carlo function    
def montecarlo(x):
    
    start_time=time.time()
    
    iteration=0.0
    
    gamma=0.05
    
    #set how often to output the energy 
    often=10
    
    timelist=[0.0]
    
    # calculate the initial energy
    energy=0.0
    for i in range(0,n):
        for j in range(i+1,n):
            distance=np.sqrt(sum((x[i]-x[j])**2))
            energy=energy+1.0/distance
    energylist=[energy]
    
    # the main loop to reduce the energy        
    while True:
    
        # randomly choose a point to move
        i=random.randint(0,n-1)
    
        # store the old coordinates of this point
        old=np.array(x[i])
    
        # randomly move this point
        x[i]=x[i]+gamma*(2.0*np.random.random(3)-1.0)
        x=proj(x,i)
    
        # calculate the difference in energy
        difference=0.0
        for j in range(0,n):
            if(j!=i):
                distance=np.sqrt(sum((x[i]-x[j])**2))
                distanceold=np.sqrt(sum((old-x[j])**2))
                difference=difference+1.0/distance-1.0/distanceold;
    
        # accept or reject the move 
        if(difference<0.0):
            energy=energy+difference
        else:
            x[i]=old
            
        if abs(difference)<0.0000001:
            break
        else:
            t=time.time()-start_time
            iteration=iteration+1
            
        ## output energy to a file
        if(iteration%often==0):
            timelist.append(t)
            energylist.append(energy)
            
    return x,energy,timelist,energylist

    
        
#gradient flow function
def gradflow(x):
    
    start_time=time.time()
    
    iteration=0.0
    
    gamma=0.05
    
    #set how often to output the energy 
    often=10
    
    timelist=[0.0]
    
    # calculate the initial energy
    energy=0.0
    for i in range(0,n):
        for j in range(i+1,n):
            distance=np.sqrt(sum((x[i]-x[j])**2))
            energy=energy+1.0/distance
            
    energylist=[energy]
    
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
            #print x
            
            # calculate the difference in energy
            difference=0.0
            for j in range(n):
                if(j!=i):
                    distance=np.sqrt(sum((x[i]-x[j])**2))
                    distanceold=np.sqrt(sum((old-x[j])**2))
                    difference=difference+1.0/distance-1.0/distanceold;
            #print difference
            
            #sum energy
            if(difference<0.0):
                energy=energy+difference
            else:
                x[i]=old
                gamma=gamma/2
                
            iteration=iteration+1
            t=time.time()-start_time
            
            if(iteration%often==0):
                timelist.append(t)
                energylist.append(energy)
            
        if abs(difference)<0.0000001:
            break
            
            
    return x,energy,timelist,energylist

#output:
X1=np.genfromtxt("random"+str(n).zfill(3))
x_mc,energy_mc,t_mc,el_mc=montecarlo(X1)
X2=np.genfromtxt("random"+str(n).zfill(3))
x_gf,energy_gf,t_gf,el_gf=gradflow(X2)

x=x_mc

#Create a sphere
theta, phi = np.mgrid[0.0:np.pi:100j, 0.0:2.0*np.pi:100j]
xs = np.sin(theta)*np.cos(phi)
ys = np.sin(theta)*np.sin(phi)
zs = np.cos(theta)

X=np.genfromtxt("random"+str(n).zfill(3))

#convert data
X1=[]
X2=[]
X3=[]
for i in range(0,n):
    X1.append(X[i,0])
    X2.append(X[i,1])
    X3.append(X[i,2])
    

#Render
fig = plt.figure()
ax = fig.add_subplot(221, projection='3d')
ax.plot_surface(xs, ys, zs,  rstride=4, cstride=4, color='yellow', alpha=0.3, linewidth=0)
ax.scatter(X1,X2,X3,color='black',s=80)
plt.axis('off')
ax.view_init(elev=0.,azim=0)

ax.set_xlim([-1.0,1.0])
ax.set_ylim([-1.0,1.0])
ax.set_zlim([-1.0,1.0])
ax.set_aspect("equal")
ax.set_title("Initial configuration")

#convert data
x1=[]
x2=[]
x3=[]
for i in range(0,n):
    x1.append(x[i,0])
    x2.append(x[i,1])
    x3.append(x[i,2])

ax = fig.add_subplot(222, projection='3d')
ax.plot_surface(xs, ys, zs,  rstride=4, cstride=4, color='yellow', alpha=0.3, linewidth=0)
ax.scatter(x1,x2,x3,color='black',s=80)
plt.axis('off')
ax.view_init(elev=0.,azim=0)

ax.set_xlim([-1.0,1.0])
ax.set_ylim([-1.0,1.0])
ax.set_zlim([-1.0,1.0])
ax.set_aspect("equal")
ax.set_title("Final Configuration")

#mc stopping point:
x1=max(t_mc)
y1=min(el_mc)

#gf stopping point
x2=max(t_gf)
y2=min(el_gf)

miny=min([y1,y2])
maxy=el_mc[0]

ax2=fig.add_subplot(212)
ax2.plot(t_mc,el_mc, 'b-', label="Monte Carlo")
ax2.plot(t_gf,el_gf, 'r-',label="Gradient Flow")
ax2.plot(x1,y1,'bx')
ax2.plot(x2,y2,'rx')
ax2.set_title("Monte Carlo vs Gradient Flow")
ax2.set_xlabel("time taken (s)")
ax2.set_ylabel("energy")
plt.legend()

plt.show()