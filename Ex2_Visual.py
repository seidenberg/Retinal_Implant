# -*- coding: utf-8 -*-
"""
Computer Simulations of Sensory Systems
Exercise 2
Retinal Implant Simulation

version: 3.4
authors: Vitaliy Banov, Savina Kim, Ephraim Seidenberg
emails: vitanov@ruri.waseda.jp, savkim@ethz.ch, sephraim@ethz.ch
date: April 29 2019
"""

import cv2
import math
import skimage
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
from tkinter import filedialog
from numpy import exp, sin, cos, pi

# variable declaration
eye_radius = 1.25
screen_height = 30
display_length = 60

# %%

# open a dialog to select the input image and store its path
image_path = filedialog.askopenfilename()
# read in the selected image with matplotlib for displaying purposes
image = plt.imread(image_path)
# read in the selected image with cv2, convert to grayscale and store in matrix form
img_g_float = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
# show the image to the user for selection of the fixation point
plt.imshow(img_g_float, cmap="gray")
# memory allocation
Zones = np.zeros(image.shape, dtype=np.uint8)

print("Please click")
# get fixation point
selected = np.round(np.array(plt.ginput(1)))

print("Clicked:", selected)

x_0 = selected[0][1]
y_0 = selected[0][0]

print(x_0)
print(y_0)
plt.close()

# %%

# store the larger of the two distances from fixation point to left and right side of the image.
x_max = max(image.shape[0] - selected[0][0], selected[0][0])
# store the larger of the two distances from fixation point to top and bottom of the image
y_max = max(image.shape[1] - selected[0][1], selected[0][1])
# compute the largest distance inside the image starting from the fixation point
largest_distance = np.sqrt(x_max ** 2 + y_max ** 2)

# print(image.shape, x_max, y_max, largest_distance)

# mapping from screen to retina
individual_zones = (largest_distance / 10) * np.arange(1, 11)
eccentricity_vector = largest_distance * (screen_height / (image.shape[0])) / np.arange(1, 11)
print('Eccentricity Vector', eccentricity_vector)
rfs = eye_radius * eccentricity_vector / display_length
print('RFS', rfs)
sigma_vector = (np.arctan(rfs / eye_radius)) * 10
sigma_vector = np.flipud(sigma_vector)
print('Sigma Vector', sigma_vector)


# %%
# DOG functions as implemented by T. Haslwanter

def gaussian2D(x, y, sigma):
    return (1.0 / (1 * math.pi * (sigma ** 2))) * math.exp(-(1.0 / (2 * (sigma ** 2))) * (x ** 2 + y ** 2))


# make matrix from function
def receptiveFieldMatrix(func):
    h = 30  # height
    g = np.zeros((h, h))  # grid
    for xi in range(0, h):
        for yi in range(0, h):
            x = xi - h / 2
            y = yi - h / 2
            g[xi, yi] = func(x, y)
    return g


def plotFilter(show):
    g = receptiveFieldMatrix(show)
    cv2.imshow('Filter', image)


def mexicanHat(x, y, sigma1):
    return gaussian2D(x, y, sigma1) - np.abs(gaussian2D(x, y, sigma1 * 1.6))


# %%
# Making Zones
Zones = np.zeros((10, image.shape[0], image.shape[1]))
print(Zones.shape)

# for i in range(len(sigma_vector)):
for i in range(len(sigma_vector)):
    print("generating zone", i)
    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
            if (np.sqrt((x - x_0) ** 2 + (y - y_0) ** 2) <= (individual_zones[i])):
                Zones[i, x, y] = 1
    # plt.imshow(Zones[i])

# %% Defining circular zones

# memory allocation
Zone_Sections = np.zeros(Zones.shape)
Zone_Sections[0] = Zones[0]
for i in range(1, 9):
    Zone_Sections[i] = Zones[i] - Zones[i - 1]

# %% Partition actual image

# memory allocation
Partitioned_Image = np.zeros(Zones.shape)
for i in range(len(Zones)):
    Partitioned_Image[i] = np.multiply(img_g_float, Zone_Sections[i])
    # plt.imshow(Partitioned_Image[i])

# %% final convolution from zones by kernel

# memory allocation
Img_FinalConvolution = np.zeros(Zones.shape)
for i in range(len(sigma_vector)):
    Img_FinalConvolution[i] = signal.convolve(Partitioned_Image[i],
                                              receptiveFieldMatrix(lambda x, y: mexicanHat(x, y, sigma_vector[i])),
                                              mode='same')
    # imgplot = plt.imshow(Img_FinalConvolution[i], cmap="gray")

final_product = np.sum(Img_FinalConvolution, axis=0)
final_product_equalized = skimage.exposure.equalize_hist(final_product)
final_product_normalized = final_product / Img_FinalConvolution.max()
# Ganglion cell FINAL OUTPUT
# cv2.imshow('ganglion cell simulation', final_product_normalized)
cv2.imshow('ganglion cell simulation equalized', final_product_equalized)
cv2.imwrite('ganglion_cell_output_equalized.jpg', final_product_equalized * 255)
cv2.imwrite('ganglion_cell_output.jpg', final_product)
print('ganglion cell output saved')

# %% Primary visual cortex simulation

# TODO: Set reasonable parameters (the ones here work well for the Door and Lena images)
# set the number of different orientations used as requested by the exercise instructions
n_orientations = 6
# set the width of the kernel (representing a square shaped receptive field) in pixels
kernel_width = 40
# define the width of the gaussian envelope (standard deviation)
std_dev = 1
# set the ellipticity of the gaussian to 1 (circular)
ellipticity = .7
# set the phase offset of the wave function (cosine factor in the Gabor function) to 0
phase_shift = 0
# set the wavelength to some reasonable value
wavelength = .5

# set the orientations of features to which the filter should be most responsive as requested in the exercise (use radians)
orientations = np.linspace(0, 150, n_orientations) * pi / 180
# place the pixels on the side of the kernel on a space from -1 to 1
kernel_side = np.linspace(-1, 1, kernel_width)
# make a square grid for the kernel, separate x and y values
x, y = np.meshgrid(kernel_side, kernel_side)
# compute all x' values to be entered into the Gabor function
x_primes = [x * cos(orientation) + y * sin(orientation) for orientation in orientations]
# compute all x' values to be entered into the Gabor function
y_primes = [-x * sin(orientation) + y * cos(orientation) for orientation in orientations]

# compute all kernels applying the Gabor function to each value and store them all into a list
kernels = [np.array(exp(-(x_prime ** 2 + ellipticity ** 2 * y_prime ** 2) / (2 * std_dev ** 2)) * cos(
    2 * pi * x_prime / wavelength + phase_shift), dtype=np.float32) for x_prime, y_prime in zip(x_primes, y_primes)]

# convert the image to grayscale and store it in matrix form
image_sw = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
# memory allocation
image_sw_filtered = np.zeros(image_sw.shape, dtype=np.float32)
# compute how many times the side of a kernel fits into the height of the image and round it up
im_height_in_kernels = int(np.ceil(image_sw.shape[0] / kernel_width))
# compute how many times the side of a kernel fits into the width of the image and round it up
im_width_in_kernels = int(np.ceil(image_sw.shape[1] / kernel_width))

# %% debugging step for individual orientation kernels
# split up the image into kernel-sized squares
# for i in range(im_height_in_kernels):
#     for j in range(im_width_in_kernels):
#         current_block = image_sw[kernel_width*i:kernel_width*i + kernel_width, kernel_width*j:kernel_width*j + kernel_width]
#         filtered_blocks = np.array([current_block * kernel[:current_block.shape[0],:current_block.shape[1]] /255 for kernel in kernels], dtype=np.float32)
# filtered_blocks = np.array([current_block * kernel[:current_block.shape[0],:current_block.shape[1]] /255 for kernel in kernels], dtype=np.float32)
#         image_sw_filtered[kernel_width*i:kernel_width*i + kernel_width, kernel_width*j:kernel_width*j + kernel_width] = sum(filtered_blocks) / n_orientations

# %% filtering step

# memory allocation
images_sw_filtered = [np.zeros(image_sw.shape, dtype=np.float32) for i in range(n_orientations)]

# split up the image into kernel-sized squares
for i in range(im_height_in_kernels):
    for j in range(im_width_in_kernels):
        current_block = image_sw[kernel_width * i:kernel_width * i + kernel_width,
                        kernel_width * j:kernel_width * j + kernel_width]
        # filter the current block using kernel and normalize
        filtered_blocks = np.array([cv2.filter2D(current_block, cv2.CV_32F, kernel) for kernel in kernels],
                                   dtype=np.float32)
        # filtered_blocks = np.array([current_block * kernel[:current_block.shape[0],:current_block.shape[1]] /255 for kernel in kernels], dtype=np.float32)
        image_sw_filtered[kernel_width * i:kernel_width * i + kernel_width,
        kernel_width * j:kernel_width * j + kernel_width] = sum(filtered_blocks) / n_orientations

# normalize
image_sw_filtered_normalized = image_sw_filtered / 255
# for k in range(n_orientations):
#     images_sw_filtered[k][kernel_width*i:kernel_width*i + kernel_width, kernel_width*j:kernel_width*j + kernel_width] = filtered_blocks[k]

# set the output height for the final display
cropped_height = 1000
# print(int(cropped_height / image_sw_filtered.shape[1] * image_sw_filtered.shape[0]))


# show the filtered image
cv2.imshow('v1 simulation', cv2.resize(image_sw_filtered_normalized,
                                       (cropped_height, int(cropped_height / image_sw.shape[1] * image_sw.shape[0]))))

# store the filtered image
cv2.imwrite('v1_output.jpg', image_sw_filtered)
print('stored v1 output')

# %% debugging step

# for i in range(n_orientations):
#     cv2.imshow(str(i*30) + ' degrees', cv2.resize(images_sw_filtered[i], (cropped_height, int(cropped_height / image_sw.shape[1] * image_sw.shape[0]))))
#     cv2.waitKey()

# filtered = cv2.filter2D(image_sw, cv2.CV_32F, kernels[2])


# cv2.imshow('test', filtered)
# filter the image with every kernel and store the results in a list
# filtered = [cv2.filter2D(image_sw, cv2.CV_32F, kernel) for kernel in kernels]


# cv2.imshow(sum(filtered))

# gabor function: exp
# def remotest_corner(point, shape):
#     x_ratio=point/shape[0]
#     y_ratio=point/shape[1]
#     if point/shape[1]>.5:
#         if point
#
# print(fixation_point, ' selected')
#
# image_gray = image[:,:,0]
#
# plt.set_cmap('gray')
# plt.imshow(image_gray)
# image_gray.shape

# image_adjusted = image.copy()
# image_adjusted[200:220, 980:1000] = 160
# image_adjusted[980:1000, 200:220] = 255
# plt.imshow(image_adjusted)
