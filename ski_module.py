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
from scipy import linspace, meshgrid

class Gauss():

    A = 200
    C = -1
    sigmaX = 16
    sigmaY = 16
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

    kx = 1
    ky = 1

    def __init__(self, X0, Y0):
        self.X0 = X0
        self.Y0 = Y0

    def fun(self, X, Y):
        return self.kx*(X-self.X0)**2 + self.ky*(Y-self.Y0)**2

    def fun_x(self, X, Y):
        return 2*self.kx*(X - self.X0)

    def fun_y(self, X, Y):
        return 2*self.ky*(Y - self.Y0)

class Cone():

    k = 100

    def __init__(self, X0, Y0):
        self.X0 = X0
        self.Y0 = Y0

    def fun(self, X, Y):
        return numpy.sqrt(self.k * ((X - self.X0)**2 + (Y - self.Y0)**2))

    def fun_x(self, X, Y):
        return self.k * (X - self.X0) / numpy.sqrt(self.k * ((X - self.X0)**2 + (Y - self.Y0)**2))

    def fun_y(self, X, Y):
        return self.k * (Y - self.Y0) / numpy.sqrt(self.k * ((X - self.X0)**2 + (Y - self.Y0)**2))

# added gaussian curves
points = [(65, 35), (30, 30), (120, 50), (50, 100)] 

class ski_path():

    curve_incline = 1
    gaussian_amplitude = 1
    gaussian_spread = 1

    def __init__(curve_type="cone", points):
        self.type = curve_type
        if (curve_type == "cone"):
            self.fun_type = Cone
        elif (curve_type == "Parable"):
            self.fun_type = Paraboloid
        else:
            raise Exception("Please enter a valid base curve type")

        try:
            iter(points)
        except:
            raise Exception("points need to be a list of tuples [(x, y), ...]")

    def config(curve_incline, 
                gaussian_amplitude, 
                gaussian_spread):
        self.curve_incline = curve_incline
        self.gaussian_amplitude = gaussian_amplitude
        self.gaussian_spread = gaussian_spread

    def fun(X, Y):
        cone = self.curve_type(75, 65)
        z = cone.fun(X, Y)
        for p in points:
            gau = Gauss(p[0], p[1])
            z += gau.fun(X, Y)
        return z

    def derivative(X, Y):
        cone = Cone(75, 65)
        zx = -cone.fun_x(X, Y)
        zy = -cone.fun_y(X, Y)
        for p in points:
            gau = Gauss(p[0], p[1])
            zx += gau.fun_x(X, Y)
            zy += gau.fun_y(X, Y)

        return math.atan2(zy, zx)

    def derive_path(points, org, des):
        k = 1
        probe = org
        pl = [probe]
        i = 0
        while vec_dist(probe, des) > k * 2 and i < 300:
            a = self.derivative(probe[0], probe[1])
            probe = (probe[0] + math.cos(a) * k, probe[1] + math.sin(a) * k)
            pl += [probe]
            i += 1

        return pl

    def _probe_locations(probe, k, n):
        return [(probe[0]+k*(math.cos(360/i)), probe[1]+k*(math.sen(360/i))) 
                    for i in range(1, n, 360/n)]

    def probe_path(points):
        k = 1
        probe = org
        pl = [probe]
        i = 0
        while vec_dist(probe, des) > k * 2 and i < 1000:
            probe_locations = self._probe_locations(probe, k, 8)
            probe_results = [self.fun(p[0], p[1]) for p in probe_locations]
            val, idx = min((val, idx) for (idx, val) in enumerate(probe_results))
            probe = probe_locations[idx]
            pl += [probe]
            i += 1
        return pl
