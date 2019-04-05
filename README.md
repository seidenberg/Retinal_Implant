## VIT's Glorious Advice

# OpenCV vs Scikit-Image vs Scipy vs ITK

# Ganglion Cell Filtering (for Savina)
[IPythonCodetoSteal](https://github.com/thomas-haslwanter/CSS_ipynb/blob/master/ImgProc_3_GanglionCell.ipynb)


* The notes use 1px & 2px for the definition of the Mexican Hat filters, while the assignment requires us to use actual math

(from Assignment):
Assume that  the display has a resolution (for those 30 cm) of 1400 pixels,
    and is viewed at a distance of 60 centimeter (see Figure below),
    and that the radius of the eye is typically 1.25 cm.
This lets you convert pixel location to retinal location.

* This means that the  
> edges = gaussian(img, 1) - gaussian(img, 2)

way needs to be done with real numbers. 
So the pipeline prolly needs to look like

- Import Image (such difficulty)
- MAP image FROM 1400px = 30cm @ 60cm away TO 1400px = 2pi * 1.25cm (and flipped upside down)
- Generate DoG Filters such that the mexican hats at the edge/periphery are much bigger(and thus blurrier) than the DoGs in the center from

> (side_length=10∗σ1)

 WHERE σ1 depends on distance from center Due to 

> RFS[arcmin]=10∗eccentricity[mm].

Also known as "how do I get the arc length of a circle"

- Convolve the DoG filterbank along the original image (such difficulty) 

# V1 cell processing (for Eph)
[Scikit Image section on Gabor Filters](http://scikit-image.org/docs/stable/auto_examples/features_detection/plot_gabors_from_astronaut.html#sphx-glr-auto-examples-features-detection-plot-gabors-from-astronaut-py)

- Run his gabor_demo.py which does the assignment for you (also pip install opencv-python) for it to actually work

- I have literally nothing else to say because the scikit-image link GIVES YOU the gabor filterbanks
- If in openCV then [OpenCV documentation of gabor filters](https://docs.opencv.org/3.0-beta/modules/imgproc/doc/filtering.html#getgaborkernel) you'll have to make your own filter bank

- Break the image up into X by X squares ~~technically circles that can overlap~~ 
> ((RFsize=0.072⋅RFeccentricity+0.017) as reflected in the measurements of Hubel and Wiesel) 

- Copy and paste the code that convolves each square by its "best approximation" of the gabor filter
