# POINTS ON A DISK - One Point Algorithm.

# Numerical outputs: n, minimum energy.

import matplotlib.pyplot as plt
import numpy as np

# assume known configuration for n=2   
X=np.array([[-0.662750, -0.748841],[0.662750, 0.748841]])


def norm(x):
    sumsq=0.0
    for k in range(2):
        sumsq=sumsq+(x[k]**2)
    return np.sqrt(sumsq)

def proj(x,j):
    if(j==n):
        for i in range(0,n):
            norm=np.sqrt(sum(x[i]**2))
            x[i]=x[i]/norm
    else:
        norm=np.sqrt(sum(x[j]**2))
        x[j]=x[j]/norm
    return x

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
            
        if gamma<0.000000001:
            break
        else:
            iteration=iteration+1
        
    return x,energy

for n in range(2,15):

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
    
    #find out number of rings
    numr=len(r)
    
    #determine whether or not there is a central point in previous config
    #if no: try a point on each ring and one at the centre
    if r[0]>0.05:
        #one at centre
        ptguess=np.zeros((numr+1,2))
        #one on each ring
        for i in range(numr):
            theta=np.random.uniform(0,2*np.pi)
            xs = r[i]*np.cos(theta)
            ys = r[i]*np.sin(theta)
            ptguess[i]=[xs,ys]
    #if yes: try a point on each ring (excluding centre) and one midway between centre and first ring
    else:
        ptguess=np.zeros((numr,2))
        #one at midway point:
        rmid=(r[1]-r[0])/2
        theta=np.random.uniform(0,2*np.pi)
        xs = rmid*np.cos(theta)
        ys = rmid*np.sin(theta)
        ptguess[0]=[xs,ys]
        #one on each ring excluding the centre point:
        for i in range(1,numr):
            theta=np.random.uniform(0,2*np.pi)
            xs = r[i]*np.cos(theta)
            ys = r[i]*np.sin(theta)
            ptguess[i]=[xs,ys]
    
    n=n+1
    
    #now try placing each of these points in old configuration and relax them:
    xs=[]
    energies=[]
    for i in range(len(ptguess)):
        guess=[ptguess[i]]
        Xguess=np.concatenate((X,guess),axis=0)
        x,energy=relax(Xguess)
        xs.append(x)
        energies.append(energy)
    
    #order the energies and configs from smallest to largest
    energies=np.array(energies)
    xs=np.array(xs)
    inds2=energies.argsort()
    
    sortenergies=energies[inds2]
    
    #return n, and lowest energy.
    print n
    print sortenergies[0]
    
    #choose X to be config with lowest energy
    sortxs=xs[inds2]
    X=sortxs[0]
    
    #return to beginning of loop
    
 
    

    

# #Render
# fig = plt.figure()
# ax = fig.add_subplot(111,aspect='equal')
# ax.set_xlim([-1.5,1.5])
# ax.set_ylim([-1.5,1.5])
# ax.set_aspect("equal")
# ax.set_title("{0} ".format(n)+"points on a disk")
# plt.axis('off')
# 
# #Create mulitple disks with radii from r
# if r[0]<0.01:
#     ax.scatter(0,0,color='black',s=20)
#     
# for i in range(len(r)):
#     theta= np.linspace(0,np.pi*2,100)
#     xs = r[i]*np.cos(theta)
#     ys = r[i]*np.sin(theta)
#     ax.plot(xs, ys, color='black', linewidth=2)
#     plt.axis('off')
# 
# plt.show() 