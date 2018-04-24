# CENTRAL CONFIGURATIONS - Genetic Algorithm

# Numerical output: N, generations, minimum energy, energies of metastable states.
# Graphical output: Points in ball.
# Save to file: coordinates of lowest energy config.

import numpy as np
import matplotlib.pyplot as plt
import random
from mpl_toolkits.mplot3d import Axes3D


# set the number of points
n=10

#function to calculate norm of a vector

def norm(x):
    sumsq=0.0
    for k in range(3):
        sumsq=sumsq+(x[k]**2)
    return np.sqrt(sumsq)

def relaxcc(x):
    
    # calculate the initial energy
    energy=0.0
    for i in range(0,n):
        energyp1=0.0
        for j in range(i+1,n):
            energyp1=energyp1+1.0/norm((x[i]-x[j]))
        energyp2=0.5*norm(x[i])**2
        energy=energy+energyp1+energyp2    
       
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
            
    return x,energy 
        
def half(x):
    lw=[]
    up=[]
    z=[]
    for i in range(n):
        z.append(x[i,2])
    srtz=np.sort(z)
    index=int(n/2.0)
    midvalue=srtz[index-1]
    for i in range(n):
        if x[i,2]<=midvalue:
            lw.append(x[i])
        else:
            up.append(x[i])
    return lw,up

#function to make unit vectors:
def unit(x):
    norm=np.sqrt(sum(x**2))
    xunit=x/norm
    return xunit

#cross product function:
def cross(a, b):
    prod = [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]]
    return prod
        
#function to randomly rotate points:
def randomrot(X):
    random.seed()
    u=unit(np.random.random(3))
    v=unit(np.random.random(3))
    vtil=unit((v-sum(u*v)*u))
    w=cross(u,vtil)
    M=np.array([[u[0],vtil[0],w[0]],[u[1],vtil[1],w[1]],[u[2],vtil[2],w[2]]])
    rotX=np.dot(X,M)
    return rotX
    

#function to combine 4 to make 16
def comb(X):
    l1=[]
    l2=[]
    for i in range(4):
        Xirot=randomrot(X[i])
        l1.append(half(Xirot)[0]) #will end with list of 4 sets of lower halfs
        l2.append(half(Xirot)[1]) #upper halfs
    X16=[]
    for g in range(4):
            for h in range(4):
                xnew=np.concatenate((l1[g],l2[h]),axis=0)
                X16.append(xnew)
    return X16
        
    
#start with four and relax them
X4=[]
for i in range(4):
    random.seed()
    xrand=2.0*np.random.random((n,3))-1.0
    xi=relaxcc(xrand)[0]
    X4.append(xi)


#loop over a given number of generations of genetic algorithm
#stop if minimum energy minima of generation is same as previous generation or if there arent four different minima
maxgenerations=3
oldmin=0.0
generation=1

for loops in range(maxgenerations):
    
    #Make 16    
    X16=comb(X4)
    
    #relax all 16
    energies16=[]
    X16relax=[]
    for i in range(16):
        x=X16[i]
        xirelax,energyi=relaxcc(x)
        X16relax.append(xirelax)
        energies16.append(energyi)
    
    
    #find the (different) minima
    
    energies161=np.array(energies16)
    X16relax=np.array(X16relax)
    
    indexsort=energies161.argsort()
    sortenergies=energies161[indexsort]
    sortconfigs=X16relax[indexsort]
    
    finalenergies=[sortenergies[0]]
    finalconfigs=[sortconfigs[0]]
    
    for i in range(1,16):
        if sortenergies[i]-sortenergies[i-1]>0.0001:
            finalenergies.append(sortenergies[i])
            finalconfigs.append(sortconfigs[i])
    
    if abs(oldmin-finalenergies[0])<0.00001:
        break
    else:
        if len(finalenergies)<4:
            break
        else:
            X4=finalconfigs[0:4]
            oldmin=finalenergies[0]
            generation=generation+1
            

print ("{0} {1} {2} {3}\n".format(n,generation,finalenergies[0],finalenergies))

#save the coordinates of the lowest energy configuration
x=finalconfigs[0]

filename="c-config"+str(n).zfill(3)
points=open(filename,'w')
for i in range(0,n):
    for j in range(0,3):
        points.write("{0:.6f} ".format(x[i,j]))
    points.write('\n')              
points.close()

r=max(norm(x[i]) for i in range(n))

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