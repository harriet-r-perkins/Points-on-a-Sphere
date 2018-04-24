# Points-on-a-Sphere
Third Year Project,  Mathematics at Durham University, 2017-18.

This project looks at the problem of minimising the total energy of points, interacting via an inverse square force, on the surface of the unit sphere. This is commonly refered to as Thomson's problem.
Other related minimisation topics are also investigated, included minimising the energy of points interacting under the same force on a disk and finding the central configurations of points acting under two-body central forces.

Guide to the code:

Sphere - 
1. MC.py : minimisation via the Monte Carlo method
2. GF.py : minimisation via the method OF Gradient Flow
3. MCvsGF.py : comparison of the two methods
4. Symm.py : symmetries of the points
5. Voronoi.py : voronoi construction of the points
6. EnergyApprox.py : energy approximation/fit to a curve
7. GA.py : minimisation via Genetic Algorithm

Disk - 
1. GF.1.py : minimisation via Gradient Flow, initial configuration found via Method 1: random
2. GF.2.py : minimisation via Gradient Flow, initial configuration found via Method 2: sample from distribution
3. GF.3.py : minimisation via Gradient Flow, initial configuration found via Method 3: placed in smaller disk
4. Symm.py : symmetries of the points
5. Voronoi.py : voronoi construction of the points
6. OPA.py : minimisation loop using the One Point Algorithm
7. GA.py : minimisation via Genetic Algorithm

Ball - 
1. GF.py: minimisation via Gradient Flow
2. GA.py : minimisation via Genetic Algorithm
3. ShellSort.py : shell structure of the configurations
4. Radii.py : radii of points within each central configuration - demonstration of subshells.



