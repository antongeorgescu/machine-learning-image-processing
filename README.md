# Machine Learning Image Processing - A Practical Approach

### What is Image Processing?
Image processing is a method to perform some operations on an image, in order to 
* get an enhanced image or 
* to extract some useful information from it

It is a type of signal processing in which 
* input is an image and 
* output may be image or characteristics/features associated with that image<br/>

In machine learning, image processing is an essential part of “data preparation”

### Why do we need image processing?
There are a few hot areas where imiage processing is in its element:

* **Computer Vision** - related to theory for design of artificial systems that can acquire information from images. Examples: an industrial robot, an autonomous vehicle etc. 
* **Face Detection** - important facial features are detected and else are ignored. Face detection can be treated as a specific case of object class detection.
* **Biomedical Image Enhancement & Analysis** - very important for biomedical image diagnosis, the aim of this area is to enhance the biomedical images. 
In addition to originally digital methods, such as Computed Tomography (CT) or Magnetic Resonance Imaging (MRI), initially analog imaging modalities such as traditional applications like endoscopy or radiography are nowadays equipped with digital sensors. 
* **Biometric Verification** - It refers to the automatic identification or recognition of humans by their behaviors or characteristics 
* **Signature Recognition** - an important application, which is to decide, whether a signature belongs to a given signer based on the image of signature and a few sample images of the original signatures of the signer. 
* **Character Recognition** - usually known as optical character recognition or abbreviated as OCR. 
An important subset is document content pre-processing that precedes the Natural Language Processing (NLP) algorithms

### How does image processing work?
Includes the following three steps:
* import the image via image acquisition tools
* analyze and manipulate the image
* output the altered image or provide report based on image analysis

Two types of methods used for image processing: 
* analogue image processing 
  - can be used for the hard copies like printouts and photographs
* digital image processing 
  - manipulation of the digital images by using computers
  - require three general phases: 1) pre-processing; 2) enhancement; 3) display; 

### Image processing samples
The following 2 scripts are providing concrete examples of image processing iplemented in Python:
* Sample #1: De-noise and sharpen the image by using frequency domain transformations (filters)
Run script **Frequency_Domain_Image_Tester.py** 
This script shows actual usage of 3 types of filters (Butterworth, Gaussian, and Ideal) to apply smoothing and sharpening to an original image
* Sample #2: Digit 0-9 recognition with classification model (KNN)
Run script **MNIST_Digit_Image_Tester.py**
This script shows an actual implementation of manually written digits between 0-9 recognition by using a trained classification algorithm built with nearest neighbour (KNN) 
* Supporting documentation can be found at [Image Processing for and with Machine Learning](https://github.com/antongeorgescu/machine-learning-documentation/blob/master/Image%20Processing%20with%20Machine%20Learning%20-%20Presentation.pptx) presentation
