# Leaf-Image-Classification

## About
Identification of plants by analyzing their leaves using digital image processing techniques which involves examining and extracting various morphological features such as shape, color, and texture. This approach influences technology to automate and streamline the identification process.

## Overview

This project aims to explore various classification techniques for plant leaf identification. A dataset of **874 plant species, with 250 images per category**, was collected. The goal was to classify these plant categories based on leaf images. The paper discusses different plant recognition and classification methods, comparing their implementation and performance. Six classical machine learning algorithms, including **Artificial Neural Network (ANN), Decision Tree (DT), Gaussian Naïve Bayes (GNB), K-Nearest Neighbors (KNN), Multi-Layer Perceptron (MLP), and Support Vector Machine (SVM)** were employed for classification.
The assessment of the performance of these classifiers was evaluated by extracting morphological features such as physiological length, physiological width, centroid, area, diameter, perimeter, rectangularity, rectangularity, sphericity, aspect ratio, area convexity, circularity, along with colour-based and texture-based features. The primary goal was to identify an optimal and effective machine-learning strategy for classifying leaf images.

## Morphology

The morphology of a leaf refers to its external physical structure, shape, color and various characteristics.

* **Shape:** The shape of a plant's leaves is an essential characteristic that can help in distinguishing different plant species. Leaves come in various forms, including round, oval, lanceolate, palmate, and lobed. By analyzing the shape of a leaf captured in a digital image, it becomes possible to compare it with a database of known leaf shapes to identify the plant species.

* **Color:** The color of a plant's leaves can vary significantly and is another vital feature for identification. Leaves can be green, yellow, red, purple, or a combination of colors. Digital image processing techniques enable the extraction of color information from leaf images, allowing for color-based analysis and comparison with reference data. This helps in narrowing down the potential plant species.

* **Texture:** The texture of a leaf refers to its surface characteristics, such as smoothness, roughness, veins, or patterns. Texture features extracted from digital leaf images can be analyzed to identify unique patterns or textures associated with specific plant species. This information can be compared with a database of known leaf textures to aid in the identification process.

By combining these three features - shape, color, and texture through digital image processing techniques, it becomes possible to develop algorithms and systems that can automatically identify plants based on leaf characteristics.

## Dependencies

* [Numpy](http://www.numpy.org)
* [Pandas](https://pandas.pydata.org)
* [OpenCV](https://opencv.org)
* [Matplotlib](https://matplotlib.org)
* [Scikit Learn](http://scikit-learn.org/)
* [Mahotas](http://mahotas.readthedocs.io/en/latest/)

It is recommended to use [Visual Studio Code (version 1.81)](https://code.visualstudio.com/) and use a `Jupyter Notebook`.

## Project structure

* [FeatureExtraction.ipynb](FeatureExtraction.ipynb): It contains the create_dataset() function which performs image pre-processing and feature extraction on the dataset (extract features from each image within a directory containing multiple leaf images). The ultimate goal is to store the extracted features in a CSV file.

* [ClassifyLeaves.ipynb](ClassifyLeaves.ipynb): Using the extracted features as inputs to the model and classifying them using various ML classifiers which includes *Artificial Neural Network (ANN), Decision Tree (DT), Gaussian Naïve Bayes (GNB), K-Nearest Neighbors (KNN), Multi-Layer Perceptron (MLP), and Support Vector Machine (SVM)*. 

## Methodology

### 1. Pre-processing of images

The following steps were followed for pre-processing the image:

  1. Resizing of images into 224×224 pixels.
  2. Conversion of RGB image into Grayscale image.
  3. Conversion of the Grayscale image into a Binary image and smoothing image using the Gaussian filter.
  4. Closing of holes using Morphological Transformation.
  5. Boundary extraction using contours.

### 2. Feature extraction

Various types of leaf features were extracted from the pre-processed image which are listed as follows:

  1. Shape-based features: It includes features like physiological length, physiological width, area, perimeter, aspect ratio, rectangularity, circularity, convex area, convex ratio, etc.
  2. Color-based features: It includes mean and standard deviations of R,G and B channels.
  3. Texture-based features: It includes features like contrast, correlation, inverse difference moments, entropy, etc.

### 3. ML Model building and testing

  (a) Sampling and splitting of the dataset into training and testing sets with an 80:20 ratio.<br>
  (b) Features were then scaled using [StandardScaler](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html).<br>
  (c) Also parameter tuning was done to find the appropriate hyperparameters of the model using [GridSearchCV](http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html).<br>
  (d) Applying Machine Learning models across the dataset to classify the plant species.
