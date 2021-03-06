#!/usr/bin/python3

import sys
import math

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

import numpy
from numpy import linspace
from numpy.random import randn, shuffle
from scipy import linspace, meshgrid, arange, empty, concatenate, newaxis, shape

# =========================
## generating ordered data:

N = 64
# x = sorted(randn(N))
# y = sorted(randn(N))

ranx = [0, 150]
rany = [0, 130]

x = linspace(ranx[0], ranx[1], N)
y = linspace(rany[0], rany[1], N)

X, Y = meshgrid(x, y)

X0 = 0
Y0 = 0

class Gauss():

    A = 2000
    C = -1
    sigmaX = 8
    sigmaY = 8
    sigmaX2 = sigmaX**2
    sigmaY2 = sigmaY**2

    def __init__(self, X0, Y0):
        self.X0 = X0
        self.Y0 = Y0

    def fun(self, X, Y):
        x_part = (X - self.X0)**2 / self.sigmaX2
        y_part = (Y - self.Y0)**2 / self.sigmaY2
        Z = self.A * numpy.exp(self.C * (x_part + y_part))
        return Z

    def fun_x(self, X, Y):
        mult = 2*(X - self.X0) / self.sigmaX2
        x_part = (X - self.X0)**2 / self.sigmaX2
        y_part = (Y - self.Y0)**2 / self.sigmaY2
        return self.A * mult * numpy.exp(self.C * (x_part + y_part)) 

    def fun_y(self, X, Y):
        mult = 2*(Y - self.Y0) / self.sigmaY2
        x_part = (X - self.X0)**2 / self.sigmaX2
        y_part = (Y - self.Y0)**2 / self.sigmaY2
        return self.A * mult * numpy.exp(self.C * (x_part + y_part)) 

class Paraboloid():

    kx = 1/4
    ky = 1/4

    def __init__(self, X0, Y0):
        self.X0 = X0
        self.Y0 = Y0

    def fun(self, X, Y):
        return self.kx*(X-self.X0)**2 + self.ky*(Y-self.Y0)**2

    def fun_x(self, X, Y):
        return 2*self.kx*(X - self.X0)

    def fun_y(self, X, Y):
        return 2*self.ky*(Y - self.Y0)

# added gaussian curves
points = [(65, 45), (30, 30), (120, 50), (50, 100)] 


def curve(X, Y):
    parab = Paraboloid(75, 65)
    z = parab.fun(X, Y)
    for p in points:
        gau = Gauss(p[0], p[1])
        z += gau.fun(X, Y)
    return z

def vec_dist(a, b):
    return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2) 

def ski(f, org, des):
    k = 7.5
    probe = org
    pl = [probe]
    tries = k * 2
    i = 0
    while vec_dist(probe, des) > k * 2:
        probe_locations = [
            (probe[0]+k, probe[1]+k), 
            (probe[0]+k, probe[1]-k), 
            (probe[0]-k, probe[1]+k), 
            (probe[0]-k, probe[1]-k),
            (probe[0]+k, probe[1]),
            (probe[0], probe[1]+k),
            (probe[0]-k, probe[1]),
            (probe[0], probe[1]-k)]
        probe_results = [f(p[0], p[1]) for p in probe_locations]
        val, idx = min((val, idx) for (idx, val) in enumerate(probe_results))
        probe = probe_locations[idx]
        pl += [probe]
        i += 1
    return pl

# Z = base function
Z = curve(X, Y)
origin, destiny = (0, 0), (75, 65) # 0, 0 and X0, Y0
P = ski(curve, origin, destiny)

# # derivative
# Z = parab.fun_x(X, Y)
# for p in points:
#     gau = Gauss(p[0], p[1])
#     try:
#         Z += gau.fun_x(X, Y)
#     except:
#         Z = gau.fun_x(X, Y)

# Z = 2*X**2 - Y**2 # saddle
# Z = Y # plane

# ======================================
## reference picture (X, Y and Z in 2D):


def show_3D():

    line_x = [P[i][0] for i in range(len(P))]
    line_y = [P[i][1] for i in range(len(P))]
    line_z = [0 for i in range(len(P))]
    
    fig = plt.figure()
    ax = fig.add_subplot(111 , projection='3d')
    ax.set_xlim(ranx)
    ax.set_ylim(rany)
    ax.view_init(elev=90, azim=270)

    ax.plot(line_x, line_y, line_z, '-k', linewidth=5)

    surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.jet, linewidth=0)

    title = ax.set_title("SKI")
    title.set_y(1.01)

    # ax.xaxis.set_major_locator(MaxNLocator(20))
    # ax.yaxis.set_major_locator(MaxNLocator(20))
    # ax.zaxis.set_major_locator(MaxNLocator(10))

    fig.tight_layout()
    fig.savefig('3D-constructing-{}.png'.format(N))

def show_2D():

    line_x = [P[i][0]*130/300 for i in range(len(P))]
    line_y = [P[i][1]*150/300 for i in range(len(P))]
    
    plt.plot(line_x, line_y)
    plt.imshow(Z, cmap='viridis', aspect=130/150, origin='lower')
    plt.colorbar()
    plt.tight_layout()
    plt.savefig("Heatmap.png")

# show_3D()
show_2D()
