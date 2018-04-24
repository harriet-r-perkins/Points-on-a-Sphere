# POINTS ON A SPHERE - Approximating the Energies

# Graphical output: Plot of the fitted curve W(N), Plot of the deviations of true energies from the fitted values E(N)-W(N).

import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as optimization

dataset=np.genfromtxt("finalenergies") #input: needs energies of relaxed configurations (for n=2-30 points)
data=dataset[:,2]
#for example, taking values found by looping over 'Sphere - GF':
data=np.array([   0.5,         1.732051,    3.674235,    6.474691,    9.985281,   14.452977,
   19.675288,   25.759987,   32.716949,   40.596451,   49.165253,   58.853231,
   69.306363,   80.670244,   92.911655,  106.050405,  120.084467,  135.089468,
  150.881568,  167.641622,  185.287536,  203.930191,  223.347074,  243.81276,
  265.133326,  287.302615,  310.491542,  334.63444,   359.603946])

x=[i for i in range(2,31)]

guess=np.array([0.0,0.0])

#define function
def f(x,a,b):
    return 0.5*x**2*(1+a*x**(-0.5)+b*x**(-1.5))

opt=optimization.curve_fit(f,x,data,guess)[0]

aa=opt[0]
bb=opt[1]

est=[f(x[i],aa,bb) for i in range(len(x))]

#plot of curve
fig = plt.figure()
ax = fig.add_subplot(211)
ax.plot(x,est,'b-',label='W(N)')
ax.plot(x,data,'rx',label='E(N)')
ax.set_title("Estimated fit to a curve")
ax.set_ylabel("Energy")
ax.legend()

#plot of deviation of true energies from the fit
dev=data-est
zeros=[0 for n in range(2,31)]

ax2 = fig.add_subplot(212)
ax2.plot(x,dev,'bo')
ax2.plot(x,zeros,'k-')
ax2.set_xlim([2,30])
ax2.set_title("Deviation of energies from the fit")
ax2.set_xlabel("Number of charged particles, N")
ax2.set_ylabel("E(N)-W(N)")

plt.show()
