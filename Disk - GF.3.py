# POINTS ON A DISK - Gradient Flow, Method 3: init points in disk of smaller radius.

# Numerical outputs: energies of metastable states, ratio. 
# Graphical outputs: points on disk, for each metastable state.
# Save to file : coordinates of final configuration.

import numpy as np
import matplotlib.pyplot as plt
import random
from mpl_toolkits.mplot3d import Axes3D

n=10

# project one or all points to the disk boundary
def proj(x,j):
    if(j==n):
        for i in range(0,n):
            norm=np.sqrt(sum(x[i]**2))
            x[i]=x[i]/norm
    else:
        norm=np.sqrt(sum(x[j]**2))
        x[j]=x[j]/norm
    return x
    
#function to calculate norm of a vector
def norm(x):
    sumsq=0.0
    for k in range(2):
        sumsq=sumsq+(x[k]**2)
    return np.sqrt(sumsq)


def relax(x):
    
    # calculate the initial energy
    energy=0.0
    for i in range(0,n):
        for j in range(i+1,n):
            distance=np.sqrt(sum((x[i]-x[j])**2))
            energy=energy+1.0/distance    
        

    #the main loop to reduce the energy       
    iteration=0
    
    gamma=0.05
    
    while True:
    
        for i in range(n):
            
            # store the old coordinates of point i
            old=np.array(x[i])
        
            #calculate force acting on each point i
            F=np.zeros(2)
            for j in range(n):
                if j!=i:
                    dist=x[i]-x[j]
                    normdist=norm(dist)
                    F=F+(dist/(normdist**3))
                
            #move position of point i in direction of force
            x[i]=x[i]+gamma*F
            
            #if its moved outside of disk boundary - move back to boundary
            if norm(x[i])>1:
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
            
        if gamma<0.0000001:
            break
        else:
            iteration=iteration+1
        
    return x,energy



#run it many times to take an average of the different minima:

loops=100

energies=[]
xs=[]

for i in range(loops):
    
    #sampling from distribution f(r)
    rlist=[]
    for i in range(n):
        r=0.2*random.random()
        rlist.append(r)
    
    angles=np.random.uniform(0,2*np.pi,n)
    
    x=[]
    y=[]
    for i in range(n):
        x.append(rlist[i]*np.sin(angles[i]))
        y.append(rlist[i]*np.cos(angles[i]))
    
    X=np.column_stack((x,y))
    
    x,energy=relax(X)
    energies.append(energy)
    xs.append(x)
    

energies=np.array(energies)
xs=np.array(xs)

inds=energies.argsort()
sortenergies=energies[inds]
sortxs=xs[inds]

finalenergies=[sortenergies[0]]
finalxs=[sortxs[0]]
ratio=[1]
count=0
for i in range(loops-1):
    if sortenergies[i+1]-sortenergies[i]>0.01:
        finalenergies.append(sortenergies[i+1])
        finalxs.append(sortxs[i+1])
        ratio.append(1)
        count=count+1
    else:
        ratio[count]=ratio[count]+1
    
print finalenergies
tratio=[(ratio[i]/float(loops))*100 for i in range(len(ratio))]
print tratio


#save the coordinates of the lowest energy configuration
mindisk=finalxs[0]
    
filename="disk"+str(n).zfill(3)
points=open(filename,'w')
for i in range(n):
    for j in range(2):
        points.write("{0:.6f} ".format(mindisk[i,j]))
    points.write('\n')              
points.close()


#Plot:

#Create a sphere
theta= np.linspace(0,np.pi*2,100)
xs = np.cos(theta)
ys = np.sin(theta)

for k in range(len(finalxs)):
    
    x=finalxs[k]    
    
    #convert data
    x1=[]
    x2=[]
    for i in range(0,n):
        x1.append(x[i,0])
        x2.append(x[i,1])
    
        
    #Render
    fig = plt.figure()
    ax = fig.add_subplot(111,aspect='equal') 
    ax.plot(xs, ys, color='red', linewidth=0.4)
    ax.scatter(x1,x2,color="black",s=80)
    plt.axis('off')
    
    ax.set_xlim([-1.5,1.5])
    ax.set_ylim([-1.5,1.5])
    ax.set_aspect("equal")
    ax.set_title("{0} ".format(n)+"points on a disk")
    
    plt.show() 