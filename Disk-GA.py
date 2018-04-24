# POINTS ON A DISK - Genetic Algorithm

# Numerical outputs: N, generations, minimum energy, energies of metastable states.

import numpy as np
import random
import sys 

#set the number of points
n=10

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

def half(x):
    lw=[]
    up=[]
    z=[]
    for i in range(n):
        z.append(x[i,1])
    srtz=np.sort(z)
    index=int(n/2.0)
    midvalue=srtz[index-1]
    for i in range(n):
        if x[i,1]<=midvalue:
            lw.append(x[i])
        else:
            up.append(x[i])
    return lw,up

        
#function to randomly rotate points:
def randomrot(X):
    random.seed()
    theta=np.pi*np.random.random() #i.e random angle between 0 and pi
    M=np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta),np.cos(theta)]])
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


#relax 50 random configs to get a mix of shells, pick 4 distinct lowest-energy structures (if there are 4)
energies=[]
xs=[]

for i in range(50):
    
    count=0
    rand=np.zeros((n,2))
    while count<n:
        r=2.0*np.random.random(2)-1.0
        if norm(r)>1:
            continue
        else:
            rand[count]=r
            count=count+1

    x,energy=relax(rand)
    energies.append(energy)
    xs.append(x)

energies=np.array(energies)
xs=np.array(xs)

inds=energies.argsort()
sortenergies=energies[inds]
sortxs=xs[inds]

finalenergies=[sortenergies[0]]
finalxs=[sortxs[0]]
for i in range(49):
    if sortenergies[i+1]-sortenergies[i]>0.01:
        finalenergies.append(sortenergies[i+1])
        finalxs.append(sortxs[i+1])

if len(finalenergies)<4:
    sys.exit('not enough distinct minima')
else:
    X4=finalxs[0:4]

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
        xirelax,energyi=relax(x)
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
        if sortenergies[i]-sortenergies[i-1]>0.01:
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

                      
print("{0} {1} {2} {3}\n".format(n,generation,finalenergies[0],finalenergies))



    


        

                



    
    
    


    


    







