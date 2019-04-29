import numpy as np 
import matplotlib
import math


# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import skimage
from scipy import ndimage
from skimage.color import rgb2gray
import PIL
import scipy
import cv2
import os

import matplotlib.image as mpimg
import scipy.signal as signal
#%%
def displayImage(image):
    plt.imshow(image)
plt.show()

image = cv2.imread('TheDoor.jpg')
displayImage(image)

img_g_float = rgb2gray(img)
displayImage(img_g_float)
#%%
inFile = 'TheDoor.jpg'
img = plt.imread(inFile)
plt.imshow(img)
# plt.show()

Zones = np.zeros(img.shape, dtype=np.uint8)

print("Please click")
#selected = np.round(np.array(plt.ginput(1)))

# ginput(n=1,timeout=30,show_clicks=True)

print("Clicked:", selected)
#%%
x_0 = selected[0][1]
y_0 = selected[0][0]

print(x_0)
print(y_0)
plt.close()

for x in range(0,img.shape[1]):
    for y in range(1,img.shape[0]):
        if(50<(np.sqrt((x-x_0)**2+(y-y_0)**2))<100):
            Zones[x,y]=255
            # print(x)
print('finished')

plt.imshow(Zones)
plt.show()

#%%

# fixation_point = np.round(np.array(plt.ginput(1)))
# # store the larger of the two distances from fixation point to left and right side of the image.
x_max = max(image.shape[0]-selected[0][0], selected[0][0])
# # store the larger of the two distances from fixation point to top and bottom of the image
y_max = max(image.shape[1]-selected[0][1], selected[0][1])
# # compute the largest distance inside the image starting from the fixation point
largest_distance = np.sqrt(x_max**2 + y_max**2)

print(image.shape, x_max, y_max, largest_distance)

eye_radius = 1.25
screen_height = 30

individual_zones = (largest_distance/10)*np.arange(1,11)

eccentricity_vector = largest_distance*(screen_height/(img.shape[0]))/np.arange(1,11)
print('Eccentricity Vector', eccentricity_vector)

display_length = 60

rfs = eye_radius*eccentricity_vector / display_length
print('RFS',rfs)

sigma_vector = (np.arctan(rfs/eye_radius))*10
sigma_vector = np.flipud(sigma_vector)
print('Sigma Vector',sigma_vector)

#%%

def gaussian2D(x, y, sigma):
    return (1.0/(1*math.pi*(sigma**2)))*math.exp(-(1.0/(2*(sigma**2)))*(x**2 + y**2))

"""make matrix from function"""
def receptiveFieldMatrix(func):
    h = 30 # height
    g = np.zeros((h,h)) # grid
    for xi in range(0,h):
        for yi in range(0,h):
            x = xi-h/2
            y = yi-h/2
            g[xi, yi] = func(x,y)
    return g

def plotFilter(show):
    g = receptiveFieldMatrix(show) 
    plt.imshow(g, cmap="gray")
#%%
for i in range(len(sigma_vector)):
    plotFilter(lambda x,y:gaussian2D(x,y,sigma_vector[i]))
    
#%%
# Convolution is the process of applying the filter to the input, which is the image I(x,y) denoting the grey value of the pixel at the specified position.
# When applying the gaussian filter every neuron in the output layer is excited by nearby image neurons.
Img_Gaussian = signal.convolve(img_g_float, receptiveFieldMatrix(lambda x,y: gaussian2D(x,y,5)), mode='same')
imgplot = plt.imshow(Img_Gaussian, cmap="gray")

Img_Gaussian = signal.convolve(img_g_float, receptiveFieldMatrix(lambda x,y: gaussian2D(x,y,5)), mode='same')
imgplot = plt.imshow(Img_Gaussian, cmap="gray")

# Difference of Gaussians
# The mexican hat function is a difference of gaussians, which leads to an on-center, off-surround receptive field, found in retinal ganglion cells or LGN neurons. It can be seen as a basic edge detector.

def mexicanHat(x,y,sigma1): 
    return gaussian2D(x,y,sigma1) - gaussian2D(x,y,sigma1*1.6)

plotFilter(lambda x,y: mexicanHat(x,y,3))
#%% All 10 gaussians
Kernel_Matrix = np.zeros((10,30,30))
for i in range(0,10):
    Kernel_Matrix[i] = receptiveFieldMatrix(mexicanHat(x,y,sigma_vector[i]))

#%%
Img_FinalConvolution = np.zeros(Zones.shape)   
for i in range(len(sigma_vector)):
    Img_FinalConvolution[i] = signal.convolve(Partitioned_Image[i], receptiveFieldMatrix(lambda x,y: mexicanHat(x,y,sigma_vector[i])), mode='same')
    imgplot = plt.imshow(Img_FinalConvolution[i], cmap="gray")

FINAL__FUCKING_PRODUCT = np.sum(Img_FinalConvolution, axis = 0)
plt.imshow(FINAL__FUCKING_PRODUCT)
#%%

 Making Zones
#%%
Zones = np.zeros((10,img.shape[0],img.shape[1]))
print(Zones.shape)

# for i in range(len(sigma_vector)):
for i in range(len(sigma_vector)):
    print("generating zone",i)
    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            if(np.sqrt((x-x_0)**2+(y-y_0)**2)<=(individual_zones[i])):
                Zones[i,x,y]=1  
    plt.imshow(Zones[i])

#%%

Zone_Sections = np.zeros(Zones.shape)
Zone_Sections[0] = Zones[0]
for i in range(1,9):
    Zone_Sections[i] = Zones[i] - Zones[i-1]
    
#%% Parition actual image
Partitioned_Image = np.zeros(Zones.shape)
for i in range(len(Zones)):
    Partitioned_Image[i] =np.multiply( img_g_float , Zone_Sections[i])
    plt.imshow(Partitioned_Image[i])

    
#%%
# for i in range(len(sigma_vector)):
i=1   
img_zoned = img[:,:,1]*(Zones[i]-Zones[i-1])
    
plt.imshow(img_zoned)