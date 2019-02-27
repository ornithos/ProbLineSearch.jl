# ProbLineSearch
Probabilistic Line Searches for Stochastic Optimization (Mahsereci &amp; Hennig, 2017) ported from MATLAB. 

This implementation includes a Julia translation of the `BVN` routine written by [Alan Genz](http://www.math.wsu.edu/faculty/genz/homepage). This code calculates rectangular integrals of bivariate unit Gaussians with arbitrary correlation. The code has been extensively tested against the MATLAB version and appears to be correct. The code also runs about an order of magnitude faster than the original, and perhaps can be improved still.
