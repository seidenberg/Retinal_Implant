# -*- coding: utf-8 -*-
"""
Computer Simulations of Sensory Systems
Exercise 2
Retinal Implant Simulation

version: 1.0
authors: Vitaliy Banov, Savina Kim, Ephraim Seidenberg
emails: vitanov@ruri.waseda.jp, savkim@ethz.ch, sephraim@ethz.ch
date: March 29 2019
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from tkinter import filedialog
# from matplotlib.widgets import Slider
from numpy import exp, sin, cos, pi


# show only dialog
# root = tk.Tk()
# root.withdraw()

# open a dialog to select the input image and store its path
image_path = filedialog.askopenfilename()
# TODO: Uncomment interactive file selection instead of test file path
# image_path = 'C:/Users/epsei/Desktop/Studium/nsc/CSS/Exercises/Ex_Visual/Images/All_images/IMG_3545.jpg'

# read in the selected image
image = plt.imread(image_path)
# display the image so that a fixation point can be selected TODO: add caption
# plt.imshow(image)
# allow exactly one selection and store the rounded coordinates of the selected fixation point
# fixation_point = np.round(np.array(plt.ginput(1)))
# # store the larger of the two distances from fixation point to left and right side of the image.
# x_max = max(image.shape[0]-fixation_point[0][0], fixation_point[0][0])
# # store the larger of the two distances from fixation point to top and bottom of the image
# y_max = max(image.shape[1]-fixation_point[0][1], fixation_point[0][1])
# # compute the largest distance inside the image starting from the fixation point
# largest_distance = np.sqrt(x_max**2 + y_max**2)

# print(image.shape, x_max, y_max, largest_distance)

# TODO: Set reasonable parameters or implement interactive mode to change parameters online
# set the number of different orientations used as requested by the exercise instructions
n_orientations = 6
# set the width of the kernel (representing a square shaped receptive field) in pixels
kernel_width = 20
# define the width of the gaussian envelope (standard deviation)
std_dev = 0.5
# set the ellipticity of the gaussian to 1 (circular)
ellipticity = 1
# set the phase offset of the wave function (cosine factor in the Gabor function) to 0
phase_shift = 0
# set the wavelength to some reasonable value
wavelength = 1


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
kernels = [np.array(exp(-(x_prime ** 2 + ellipticity ** 2 * y_prime ** 2) / (2 * std_dev ** 2)) * cos(2 * pi * x_prime / wavelength + phase_shift), dtype=np.float32) for x_prime, y_prime in zip(x_primes, y_primes)]

# convert the image to grayscale and store it in matrix form
image_sw = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
image_sw_filtered = np.zeros(image_sw.shape, dtype=np.float32)
# compute how many times the side of a kernel fits into the height of the image and round it up
im_height_in_kernels = int(np.ceil(image_sw.shape[0]/kernel_width))
# compute how many times the side of a kernel fits into the width of the image and round it up
im_width_in_kernels = int(np.ceil(image_sw.shape[1]/kernel_width))
# split up the image into kernel-sized squares
for i in range(im_height_in_kernels):
    for j in range(im_width_in_kernels):
        current_block = image_sw[kernel_width*i:kernel_width*i + kernel_width, kernel_width*j:kernel_width*j + kernel_width]
        filtered_blocks = np.array([current_block * kernel[:current_block.shape[0],:current_block.shape[1]] /255 for kernel in kernels], dtype=np.float32)
        image_sw_filtered[kernel_width*i:kernel_width*i + kernel_width, kernel_width*j:kernel_width*j + kernel_width] = sum(filtered_blocks) / n_orientations

cv2.imshow('v1 simulation', image_sw_filtered)
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
