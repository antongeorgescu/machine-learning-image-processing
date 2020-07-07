# The MNIST dataset provided in a easy-to-use CSV format
# The original dataset is in a format that is difficult for beginners to use. This dataset uses the work of Joseph Redmon to provide the MNIST dataset in a CSV format.
#
# The dataset consists of two files, both saved in ./datasets folder:
# mnist_train.csv
# mnist_test.csv
#
# The mnist_train.csv file contains the 60,000 training examples and labels. 
# The mnist_test.csv contains 10,000 test examples and labels. 
# Each row consists of 785 values: the first value is the label (a number from 0 to 9)
# and the remaining 784 values are the pixel values (a number from 0 to 255).

import pandas as pd
import numpy as np
from PIL import Image
from sklearn.preprocessing import MinMaxScaler

MNIST_TEST_PATH = "./datasets/mnist28x28_test.csv"
MNIST_TEST_PATH_2 = "./datasets/mnist28x28_test_2.csv"
MAX_MISSED_VALUES_PER_TEST_BATCH = 1000000  # out of 7,840,000,000 pixels
MIN_MISSED_VALUES_PER_DIGIT = 120
SELECTED_DIGITS = 10
# =======================================================================================

# read test data from external .csv file
mnist_test_df = pd.read_csv(MNIST_TEST_PATH,sep=',').astype("float")
COUNT_TEST_ROWS = mnist_test_df.values[:,1].size

# apply MAR (missing-at-random) values in the test data
empty_array = (COUNT_TEST_ROWS)
mar_rows = np.zeros(empty_array).astype("int")
print(mar_rows.shape)
for i in range(MAX_MISSED_VALUES_PER_TEST_BATCH):
    j = np.random.randint(0,COUNT_TEST_ROWS-1)
    k = np.random.randint(1,28*28)
    if (mnist_test_df.iat[j,k] == np.nan):
        i = i - 1
    else:
        mnist_test_df.iat[j,k] = np.nan
        mar_rows[j] = mar_rows[j] + 1
list_doped_images = np.argwhere(mar_rows > MIN_MISSED_VALUES_PER_DIGIT)
print(list_doped_images)

# save in "test2" .csv file for visualization
mnist_test_df2 = pd.DataFrame(mnist_test_df)
mnist_test_df2.fillna(-1.0,inplace=True)
mnist_test_df2.to_csv(MNIST_TEST_PATH_2)
print(mnist_test_df2)

# create a random selection of 10 images with #missed values > MIN_MISSED_VALUES_PER_DIGIT
flat_list_doped_images = []
for l in list_doped_images.tolist():
    flat_list_doped_images += l
list_doped_images_selected = np.random.choice(flat_list_doped_images,SELECTED_DIGITS,replace=False)
print(list_doped_images_selected)

# save the selected SELECTED_DIGITS digit original images in .png files
for i in list_doped_images_selected:
    a = np.copy(mnist_test_df.values[i])
    digit = str(int(a[0]))
    a = np.delete(a,0)
    nslices = 28
    a = np.reshape(a,(nslices, -1))
    im = Image.fromarray(a).convert('RGB')
    pix = im.load()
    rows, cols = im.size
    for x in range(cols):
        for y in range(rows):
            pix[x,y] = (int(a[y,x] // 256 // 256 % 256),int(a[y,x] // 256  % 256),int(a[y,x] % 256))
    FILENM = f'./converted/digit_orig_{digit}_{str(i)}.jpeg' 
    im.save(FILENM) 

# save the selected 10 digit "doped"" images in .png files

# run the "digit image predictor" with KNN on selected original digit images

# fix MAR by applying Python's library PyImpute methods and save the first 10 gigits in .png files 

# save fixed "test" data in .csv file

# re-run the "digit image predictor" with KNN - observe the accuracy degradation
# repeat the tests for various "missed values doping"

