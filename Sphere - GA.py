# POINTS ON A SPHERE - Genetic Algorithm

# Numerical outputs: N, generations, minimum energy, energies of metastable states.
# Save to file: configurations and their energies, found at each generation

import numpy as np
import random

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
    for k in range(3):
        sumsq=sumsq+(x[k]**2)
    return np.sqrt(sumsq)

#gradient flow relaxing function
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
            
    return x,energy


#function to halve the structures
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
    xrand=proj((2.0*np.random.random((n,3))-1.0),n)
    xi=relax(xrand)[0]
    X4.append(xi)

#loop over a given number of generations of genetic algorithm
#stop if minimum energy minima of generation is same as previous generation or if there arent four different minima
maxgenerations=5
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
        if sortenergies[i]-sortenergies[i-1]>0.0001:
            finalenergies.append(sortenergies[i])
            finalconfigs.append(sortconfigs[i])


    #save energies for this generation
    filename1="ga-it"+str(loops+1)+"-energies"+str(n).zfill(3)
    ga1=open(filename1,'w')
    ga1.write("{0} \n".format(finalenergies))     
    ga1.close()
    
    #save configurations for this generation
    filename2="ga-it"+str(loops+1)+"-configs"+str(n).zfill(3)
    ga2=open(filename2,'w')
    ga2.write("{0} \n".format(finalconfigs))
    ga2.close() 
    
    #check stopping criteria
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
         




    


        

                



    
    
    


    


    







