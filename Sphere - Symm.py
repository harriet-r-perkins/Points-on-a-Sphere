# POINTS ON A SPHERE - Symmetries

# Text output: symmetry type.
# Graphical outputs: points on sphere - viewed down main axis of symmetry (if determined)

import numpy as np
import matplotlib.pyplot as plt
import random
from mpl_toolkits.mplot3d import Axes3D


# set the number of points
n=10

#call points
x=np.genfromtxt("pointsout"+str(n).zfill(3)) #input: needs coordinates of relaxed configuration (for n points).
#for example, for n=10
x=np.array([[ 0.618473, -0.310726,  0.721762],
 [-0.401059,  0.034002,  0.915421],
 [ 0.506709, -0.836481, -0.208677],
 [ 0.544458,  0.73373,   0.406455],
 [-0.0273,    0.673138, -0.739013],
 [-0.454401, -0.848354,  0.27169 ],
 [-0.506708,  0.836482,  0.208674],
 [-0.190115, -0.396415, -0.898171],
 [-0.972817, -0.026591, -0.230044],
 [ 0.882761,  0.141214, -0.448098]])

#calculating inertia matrix
Ixx=0.0
for k in range(n):
    Ixx=Ixx+x[k,1]**2+x[k,2]**2

Iyy=0.0
for k in range(n):
    Iyy=Iyy+x[k,0]**2+x[k,2]**2

Izz=0.0
for k in range(n):
    Izz=Izz+x[k,0]**2+x[k,1]**2

Ixy=0.0
for k in range(n):
    Ixy=Ixy-x[k,0]*x[k,1]
    
Ixz=0.0
for k in range(n):
    Ixz=Ixz-x[k,0]*x[k,2]
    
Iyz=0.0
for k in range(n):
    Iyz=Iyz-x[k,1]*x[k,2]
    
IM=np.array([[Ixx,Ixy,Ixz],[Ixy,Iyy,Iyz],[Ixz,Iyz,Izz]])

#calculate eigenvalues
(val,vec)= np.linalg.eig(IM)

#test for basic symmetry:
num=1
for i in range(3):
    for j in range(i+1,3):
        if abs(val[i]-val[j]) < 0.0005:
            num=num+1

if num==2:
    print 'there is at least a C3 symmetry'
else:
    if num==4:
        print 'it is a platonic solid'
    else:
        print 'at most C2 or D2 symmetry'
    
#transform the points, X=Rx, to make x, y, z the principle axes
X=np.zeros((n,3))
for i in range(n):
    X[i]=np.dot(vec.T, x[i])



#Defining some functions

#function to calculate distance between two points:
def dist(x1,x2):
    sum=0.0
    for i in range(len(x1)):
        sum=sum+(x1[i]-x2[i])**2
    return np.sqrt(sum)

#function to test closeness of sets of points
def pntmap(A,B,acc): #A and B must be matrices with the same dimensions NxM (probably gunna be nx3)
    N,M=A.shape      #acc means accuracy
    count=0.0
    for i in range(N):
        for j in range(N):
            if dist(A[i],B[j])<acc:
                count=count+1
            else:
                continue
    if count==N:
        return 1
    else:
        return 0
        
#For the case of 'at least C3 symmetry' found
                      
#determine axis of rotation, and if not z axis, transform points
if num==2:
    if (val[0]-val[1])**2<0.001:
        pass
    else:
        if (val[0]-val[2])**2<0.001:
            X[:,1],X[:,2]=X[:,2],X[:,1].copy()
            val[1],val[2]=val[2],val[1]
        else:
            X[:,0],X[:,2]=X[:,2],X[:,0].copy()
            val[0],val[2]=val[2],val[0]

#rotataing shape about z axis to determine cylic symmetry       
    for k in range(12,2,-1):
        q=(2*np.pi)/k
        RotMatcyc=np.array([[np.cos(q),np.sin(q),0],[-np.sin(q),np.cos(q),0],[0,0,1]])
        #rotate points by angle q in xy-plane
        Xcyc=np.dot(X,RotMatcyc)
        ##check if Xnew points map onto X points
        if pntmap(Xcyc,X,0.05)==1:
            print "there is a C"+str(k)+" symmetry"
            break
        else:
            continue
            
#code determining whether there is dihedral symmetry
    for j in range(100):
        q=j*np.pi/100
        RotMatdih=np.array([[1,0,0],[0,-np.cos(q),-np.sin(q)],[0,np.sin(q),-np.cos(q)]])
        Xdih=np.dot(X,RotMatdih)
        if pntmap(Xdih,X,0.05)==1:
            print "furthermore there is a D"+str(k)+" symmetry"
            break
        else:
            continue
        
#Determining reflection symmetries:

#horizontal reflection symmetry:
    RefMath=np.array([[1,0,0],[0,1,0],[0,0,-1]])
    XRefh=np.dot(X,RefMath)
    if pntmap(XRefh,X,0.01)==1:
        print 'there is horizontal reflection symmetry'
    else:
        #vertical reflection symmetry
        for j in range(100):
                q=j*np.pi/100
                RefMatv=np.array([[-np.cos(q),np.sin(q),0],[np.sin(q),np.cos(q),0],[0,0,1]])
                XRefv=np.dot(X,RefMatv)
                if pntmap(XRefv,X,0.01)==1:
                    print 'there is vertical reflection symmetry'
                    break       

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
    x1.append(X[i,0])
    x2.append(X[i,1])
    x3.append(X[i,2])
    

#Render
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(xs, ys, zs,  rstride=4,color='blue', cstride=4, alpha=0.3, linewidth=0)
ax.scatter(x1,x2,x3,color="black",s=80)
plt.axis('off')
ax.view_init(azim=90.0,elev=90.0)

ax.set_xlim([-1.0,1.0])
ax.set_ylim([-1.0,1.0])
ax.set_zlim([-1.0,1.0])
ax.set_aspect("equal")
ax.set_title("{0} ".format(n)+"points on a sphere")

plt.show() 