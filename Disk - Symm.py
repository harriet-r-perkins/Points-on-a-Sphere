# POINTS ON A DISK - Symmetries

# Text output: symmetry type.

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

def dist(x1,x2):
    sum=0.0
    for i in range(len(x1)):
        sum=sum+(x1[i]-x2[i])**2
    return np.sqrt(sum)

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

for k in range(15,1,-1):
        q=(2*np.pi)/k
        RotMatcyc=np.array([[np.cos(q),-np.sin(q)],[np.sin(q),np.cos(q)]])
        Xcyc=np.dot(X,RotMatcyc)
        ##check if Xnew points map onto X points
        if pntmap(Xcyc,X,0.01)==1:
            print "there is a C"+str(k)+" symmetry"
            break
        else:
            if k==2:
                print "there is no cyclic symmetry"
            else:
                continue

#vertical reflection symmetry
for j in range(100):
        q=j*np.pi/100
        RefMatv=np.array([[np.cos(q),-np.sin(q)],[-np.sin(q),-np.cos(q)]])
        XRefv=np.dot(X,RefMatv)
        if pntmap(XRefv,X,0.05)==1:
            print 'there is reflection symmetry'
            break