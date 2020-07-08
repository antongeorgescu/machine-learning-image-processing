# https://gurus.pyimagesearch.com/lesson-sample-k-nearest-neighbor-classification/

# import the necessary packages
from __future__ import print_function
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn import datasets
from skimage import exposure
import numpy as np
import imutils
import cv2
import sklearn

from sklearn.model_selection import train_test_split
 
""" # handle older versions of sklearn
if int((sklearn.__version__).split(".")[1]) < 18:
    from sklearn.cross_validation import train_test_split
 
# otherwise we're using at lease version 0.18
else:
	from sklearn.model_selection import train_test_split """
 
# load the MNIST digits dataset
import pandas as pd
import numpy as np

MNIST_TRAINING_PATH = "./datasets/mnist28x28_train.csv"
MNIST_TEST_PATH = ""
TUNE_KNN_HYPERPARAMS = False

fExecutionSummary = None

def set_prediction_script_params(testfilepath,isknntuned,execsummaryfileobj):
	global TUNE_KNN_HYPERPARAMS
	global MNIST_TEST_PATH
	global fExecutionSummary

	MNIST_TEST_PATH = testfilepath
	TUNE_KNN_HYPERPARAMS = isknntuned
	fExecutionSummary = execsummaryfileobj

def run_prediction():
	# read training data from external .csv file
	mnist_training_data = pd.read_csv(MNIST_TRAINING_PATH,sep=',')

	mnist_training_df = pd.DataFrame(mnist_training_data)
	trainLabels = mnist_training_df['label'].tolist()
	trainData = mnist_training_df.drop('label',axis=1).values.tolist()

	# read test data from external .csv file
	mnist_test_df = pd.read_csv(MNIST_TEST_PATH,sep=',')
	testLabels = mnist_test_df['label'].tolist()
	testData = mnist_test_df.drop('label',axis=1).values.tolist()

	# now, let's take 10% of the training data and use that for validation
	(trainData, valData, trainLabels, valLabels) = train_test_split(trainData, trainLabels,
		test_size=0.1, random_state=84)
	
	# import matplotlib.pyplot as plt 
	# plt.gray() 
	# plt.matshow(digits.images[0]) 
	# plt.show() 
	
	# show the sizes of each data split
	print("training data points: {}".format(len(trainLabels)))
	print("validation data points: {}".format(len(valLabels)))
	print("testing data points: {}".format(len(testLabels)))

	fExecutionSummary.write(f"training data points: {len(trainLabels)}\r\n")
	fExecutionSummary.write(f"validation data points: {len(valLabels)}\r\n")
	fExecutionSummary.write(f"testing data points: {len(testLabels)}\r\n")

	if TUNE_KNN_HYPERPARAMS:
		# initialize the values of k for our k-Nearest Neighbor classifier along with the
		# list of accuracies for each value of k
		kVals = range(1, 30, 2)
		accuracies = []
		
		# loop over various values of `k` for the k-Nearest Neighbor classifier
		for k in range(1, 30, 2):
			# train the k-Nearest Neighbor classifier with the current value of `k`
			model = KNeighborsClassifier(n_neighbors=k)
			model.fit(trainData, trainLabels)
		
			# evaluate the model and update the accuracies list
			score = model.score(valData, valLabels)
			print("k=%d, accuracy=%.2f%%" % (k, score * 100))
			fExecutionSummary.write("k=%d, accuracy=%.2f%%\r\n" % (k, score * 100))
			accuracies.append(score)
		
		# find the value of k that has the largest accuracy
		i = int(np.argmax(accuracies))
		print("k=%d achieved highest accuracy of %.2f%% on validation data" % (kVals[i],
			accuracies[i] * 100))
		fExecutionSummary.write("k=%d achieved highest accuracy of %.2f%% on validation data\r\n" % (kVals[i],
			accuracies[i] * 100))
			
		# re-train our classifier using the best k value and predict the labels of the
		# test data
		model = KNeighborsClassifier(n_neighbors=kVals[i])
		model.fit(trainData, trainLabels)
		predictions = model.predict(testData)
	else:
		nno = 1
		model = KNeighborsClassifier(n_neighbors=nno)
		model.fit(trainData, trainLabels)
	
		# evaluate the model and update the accuracies list
		score = model.score(valData, valLabels)
		print("k=%d, accuracy=%.2f%%" % (nno, score * 100))
		fExecutionSummary.write("k=%d, accuracy=%.2f%%\r\n" % (nno, score * 100))
		model = KNeighborsClassifier(n_neighbors=nno)
		model.fit(trainData, trainLabels)
		predictions = model.predict(testData)
	
	# show a final classification report demonstrating the accuracy of the classifier
	# for each of the digits
	print(f"EVALUATION ON TESTING DATA: FILE {MNIST_TEST_PATH}")
	print(classification_report(testLabels, predictions))

	fExecutionSummary.write(f"EVALUATION ON TESTING DATA: FILE {MNIST_TEST_PATH}\r\n")
	fExecutionSummary.write(f"{classification_report(testLabels, predictions)}\r\n")
	