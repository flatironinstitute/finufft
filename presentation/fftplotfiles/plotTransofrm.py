#!/usr/bin/env python3

import csv
import numpy

filename = 'ptsOutX'
raw_data = open(filename, 'r')
reader = csv.reader(raw_data, delimiter=' ', quoting=csv.QUOTE_NONE)
x = list(reader)
xdata = numpy.array(x).astype('float')


filename = 'ptsOutY'
raw_data = open(filename, 'r')
reader = csv.reader(raw_data, delimiter=' ', quoting=csv.QUOTE_NONE)
y = list(reader)
ydata = numpy.array(y).astype('float')


filename = 'ptsOutZ'
raw_data = open(filename, 'r')
reader = csv.reader(raw_data, delimiter=' ', quoting=csv.QUOTE_NONE)
z = list(reader)
zdata = numpy.array(z).astype('float')


filename = 'ptsOutC'
raw_data = open(filename, 'r')
reader = csv.reader(raw_data, delimiter=' ', quoting=csv.QUOTE_NONE)
cReal = list(reader)
cRealdata = numpy.array(cReal).astype('float') 


filename = 'ptsOutC2'
raw_data = open(filename, 'r')
reader = csv.reader(raw_data, delimiter=' ', quoting=csv.QUOTE_NONE)
cImag = list(reader)
cImagdata = numpy.array(cImag).astype('float') 



filename = 'ptsOutF'
raw_data = open(filename, 'r')
reader = csv.reader(raw_data, delimiter=' ', quoting=csv.QUOTE_NONE)
FReal = list(reader)
FRealdata = numpy.array(FReal).astype('float') 


filename = 'ptsOutF2'
raw_data = open(filename, 'r')
reader = csv.reader(raw_data, delimiter=' ', quoting=csv.QUOTE_NONE)
FImag = list(reader)
FImagdata = numpy.array(FImag).astype('float') 


c = numpy.array(list(zip(cRealdata, cImagdata)))
F = numpy.array(list(zip(FRealdata,FImagdata)))

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.add_subplot(121, projection='3d')

ax.scatter(xdata , ydata, cRealdata)
#ax.plot_surface(xdata , ydata, c)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('weights on a nonuniform grid')

N=len(FRealdata[0])

modes = numpy.linspace(-N/2,N/2, num=10)
modex,modey = numpy.meshgrid(modes,modes)
print(modex.ravel().shape)
print(modey.ravel().shape)
print(FRealdata[0].shape)
ax2 = fig.add_subplot(122, projection='3d')
ax2.scatter(modex.ravel(),modey.ravel(),FRealdata[0])
#ax2.plot_surface(modes,modes,F)
ax2.set_title('Coefficient Weights in Fourier Space')

plt.show()
fig.savefig('Plot Of a Fourier Transform.png')
