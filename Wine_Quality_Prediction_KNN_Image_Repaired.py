import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer

MISSING_POINTS_COUNT = 300

# read attribute names from file
dflabels = pd.read_csv("./converted/wine_attribute_names.csv",names=["label"])
fields = dflabels["label"].to_list()
print(fields)

# read attribute normalized (scaled) data from image file
from PIL import Image
FILENAME="./ImageEditor/GenerateCorruptImage/Results/wine_dataset_img_repaired.png"
im=Image.open(FILENAME).convert('RGB')
pix=im.load()
w=im.size[0]
h=im.size[1]
datax = []
for i in range(h):
    rowx = []
    for j in range(w):
        grayscale = int(pix[j,i][0])
        rowx.extend([grayscale])
    datax.extend([rowx])

data = pd.DataFrame(datax, columns=fields) 
print(data)

X = data[fields]
y = pd.read_csv("./converted/wine_quality.csv",sep=',',names=["quality"])

# A Pearson correlation was used to identify which features correlate with wine quality. It looks as if higher the alcohol content the higher the quality. Lower density and volatile acidity also correlated with better quality as seen in the pairwise correlation chart the chart below. Only the top 5 correlated features were carried over to the KNN models.
correlations = data[fields].corrwith(y)
correlations.sort_values(inplace=True)

# the following fields are the 5 retained as having the highest correlations to wine quality
fields = correlations.map(abs).sort_values().iloc[-5:].index
print(fields) #prints the top two abs correlations

import matplotlib.pyplot as plt
import seaborn as sns

# The figure below shows Pearson Pairwise correlation of features to wine quality.
# Looks like alcohol and density are the most correlated with quality
ax = correlations.plot(kind='bar')
ax.set(ylim=[-1, 1], ylabel='pearson correlation')

# We will run now K-Nearest Neighbour (KNN) algorithm to create a "prediction model"
# 
# Since the data captured is in different magnitude ranges, it is generally a good idea to scale so one feature doesn’t get more influence than the other (in terms of scale).

from sklearn import svm
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
#from sklearn.model_selection import GridSearchCV

# we will split the data in training (70%) and testing (30%) whihc is the usual ratio
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=10)


from sklearn.neighbors import KNeighborsClassifier
import time

startproc = time.time()

# Instantiate KNN learning model (k=15)
knn = KNeighborsClassifier(n_neighbors=5)

# fit the model with training data
knn.fit(X_train, y_train)

# predict the wine rankings for the test data set
y_pred = knn.predict(X_test)

proctime = time.time() - startproc

print(y_pred,X_test)


from sklearn.metrics import accuracy_score

# how did our model perform?
y_test = y_test["quality"].to_numpy()
count_misclassified = (y_test != y_pred).sum()
print('Misclassified samples: {}'.format(count_misclassified))
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy: {:.2f}'.format(accuracy))

# ### About Confusion Matrix
# 
# A confusion matrix is a table that is often used to describe the performance of a classification model (or "classifier") 
# on a set of test data for which the true values are known.
# 
# Here is a set of the most basic terms, which are whole numbers (not rates):
# 1. true positives (TP): These are cases in which we predicted "yes" where it is actually "yes".
# 2. true negatives (TN): We predicted no, and it is actually "no".
# 3. false positives (FP): We predicted "yes", and was actually "no" (Also known as a "Type I error."
# 4. false negatives (FN): We predicted "no", and was actually "yes". (Also known as a "Type II error.")
#     
# This is a list of rates that are often computed from a confusion matrix for a binary classifier:
# 
# **Accuracy**: Overall, how often is the classifier correct? 
#     accuracy = (TP+TN)/total
# 
# **Misclassification Rate**: Overall, how often is it wrong? 
#     misrate = (FP+FN)/total
#     equivalent to 1 minus Accuracy
#     also known as "Error Rate"
# 
# **True Positive Rate**: When it's actually yes, how often does it predict yes? 
#     tprate = TP/actual_yes
#     also known as "Sensitivity" or "Recall"
# 
# **False Positive Rate**: When it's actually no, how often does it predict yes?
#     fprate = FP/actual_no
# 
# **True Negative Rate**: When it's actually no, how often does it predict no? 
#     tnrate = TN/actual_no
#     equivalent to 1 minus False Positive Rate
#     also known as "Specificity"
# 
# **Precision**: When it predicts yes, how often is it correct? 
#     precision = TP/predicted_yes
# 
# **Prevalence**: How often does the yes condition actually occur in our sample? 
#     prevalence = actual_yes/total
#     
# ![Confusion Matrix - Theoretical Foundation](https://raw.githubusercontent.com/antongeorgescu/machine-learning-documentation/master/images/Confusion-Matrix-2.PNG) 

from sklearn.metrics import confusion_matrix,f1_score

# Calculate the accuracy of prediction
# Get the accuracy score
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy: {:.4f}'.format(accuracy))

f1score = f1_score(y_test, y_pred, average='micro')
print('F1_Score: {:.4f}'.format(f1score))

# Get the confusion matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)
cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

sns.heatmap(cm, annot=True, fmt='.2g')
plt.title('Confusion matrix of the KNN classifier')    
plt.tight_layout()


import sys, os

summaryfile = 'ModelsFitness.txt'
nbdir = os.getcwd()
fsummary = open(f'{nbdir}\\{summaryfile}',"a") 
fsummary.write('Wine Quality Analysis with KNN and {:} missing points image repaired externally, Processing (sec):{:.4f}, Accuracy: {:.4f}, F1-Score: {:.4f}\r\n'.format(MISSING_POINTS_COUNT,proctime,accuracy,f1score))
fsummary.close() 

