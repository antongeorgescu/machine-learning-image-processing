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
MNIST_TEST_PATH_NAN = "./datasets/mnist28x28_test_nan.csv"
MNIST_TEST_PATH_FIXED = "./datasets/mnist28x28_test_fixed.csv"
MAX_MISSED_VALUES_PER_TEST_BATCH = 1000000  # out of 7,840,000,000 pixels
MIN_MISSED_VALUES_PER_DIGIT = 120
SELECTED_DIGITS = 10
# =======================================================================================

# read test data from external .csv file
mnist_test_df = pd.read_csv(MNIST_TEST_PATH,sep=',').astype("float")
mnist_test_columns = mnist_test_df.columns
COUNT_TEST_ROWS = mnist_test_df.values[:,1].size

# run the "digit image predictor" with KNN on selected original digit images (test)

# apply MAR (missing-at-random) values in the test data
mnist_test_mar_df = mnist_test_df.copy()
empty_array = (COUNT_TEST_ROWS)
mar_rows = np.zeros(empty_array).astype("int")
print(mar_rows.shape)
for i in range(MAX_MISSED_VALUES_PER_TEST_BATCH):
    j = np.random.randint(0,COUNT_TEST_ROWS-1)
    k = np.random.randint(1,28*28)
    if (mnist_test_mar_df.iat[j,k] == np.nan):
        i = i - 1
    else:
        mnist_test_mar_df.iat[j,k] = np.nan
        mar_rows[j] = mar_rows[j] + 1
list_doped_images = np.argwhere(mar_rows > MIN_MISSED_VALUES_PER_DIGIT)
print(list_doped_images)

# save in "test_nan" .csv file for visualization
mnist_test_mark_df = mnist_test_mar_df.fillna(-1.0,inplace=False)
mnist_test_mar_df.to_csv(MNIST_TEST_PATH_NAN)

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
            pix[x,y] = (int(a[y,x] % 256),int(a[y,x] // 256  % 256),int(a[y,x] // 256 // 256 % 256))
    suff = np.random.choice(np.linspace(1,9,num=9).astype("int").tolist(),4)
    suffx = np.apply_along_axis(lambda row: row.astype('|S1').tostring().decode('utf-8'),
                    axis=0,
                    arr=suff)
    FILENM = f'./converted/digit_orig_{digit}_{str(i)}.jpeg' 
    im.save(FILENM) 

# fix MAR by applying Python's library PyImpute methods and save the first 10 gigits in .png files 
from sklearn.impute import SimpleImputer
imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
imp_mean.fit(mnist_test_mar_df.values)
fixed_values = imp_mean.transform(mnist_test_mar_df.values)
mnist_test_mar_df = pd.DataFrame(data=fixed_values,columns=mnist_test_columns)

# save in "test_fixed" .csv file for persistance
mnist_test_mar_df.to_csv(MNIST_TEST_PATH_FIXED)

# save the selected 10 digit "fixed" images in .png files
for i in list_doped_images_selected:
    a = np.copy(mnist_test_mar_df.values[i])
    digit = str(int(a[0]))
    a = np.delete(a,0)
    nslices = 28
    a = np.reshape(a,(nslices, -1))
    im = Image.fromarray(a).convert('RGB')
    pix = im.load()
    rows, cols = im.size
    for x in range(cols):
        for y in range(rows):
            pix[x,y] = (int(a[y,x] % 256),int(a[y,x] // 256  % 256),int(a[y,x] // 256 // 256 % 256))
    suff = np.random.choice(np.linspace(1,9,num=9).astype("int").tolist(),4)
    suffx = np.apply_along_axis(lambda row: row.astype('|S1').tostring().decode('utf-8'),
                    axis=0,
                    arr=suff)
    FILENM = f'./converted/digit_fixed_{digit}_{str(i)}.jpeg' 
    im.save(FILENM) 

# re-run the digit image predictor with KNN on "fixed" images - observe the accuracy degradation
# (optional) repeat the tests for various "missed values doping"

