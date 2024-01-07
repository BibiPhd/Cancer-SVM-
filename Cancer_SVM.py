## CLASSIFICATION OF CANCER DATA USING SUPPORT VECTOR MACHINE 

import pandas as pd
import pylab as pl
import numpy as np
import scipy.optimize as opt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


# load cancer data as csv file 

website = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/cell_samples.csv'
data = pd.read_csv(website)
data.head()

# The ID field contains the patient identifiers. 
# The characteristics of the cell samples from each patient are contained in fields Clump to Mit. 
# The values are graded from 1 to 10, with 1 being the closest to benign.
# The Class field contains the diagnosis, as confirmed by separate medical procedures, as to whether the samples are benign (value = 2) or malignant (value = 4).

# Distribution of the classes based on Clump thickness and Uniformity of cell size:

ax = data[data['Class'] == 4][0:50].plot(kind='scatter', x='Clump', y='UnifSize', color='DarkBlue', label='malignant');
data[data['Class'] == 2][0:50].plot(kind='scatter', x='Clump', y='UnifSize', color='Yellow', label='benign', ax=ax);
plt.show()

data.dtypes

data = data[pd.to_numeric(data['BareNuc'], errors='coerce').notnull()]
data['BareNuc'] = data['BareNuc'].astype('int')

features = data[['Clump', 'UnifSize', 'UnifShape', 'MargAdh', 'SingEpiSize', 'BareNuc', 'BlandChrom', 'NormNucl', 'Mit']]
X = np.asarray(features)  # dataset 

y = np.asarray(data['Class']) # target: Class is already an int, so it does not require any change

## TRAIN/TEST SPLIT 

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=4)
print ('Train set:', X_train.shape,  y_train.shape)
print ('Test set:', X_test.shape,  y_test.shape)

## MODELING USING SVM
# The SVM algorithm offers a choice of kernel functions for performing its processing. 
# Basically, mapping data into a higher dimensional space is called kernelling. 
# The mathematical function used for the transformation is known as the kernel function, and can be of different types, such as:

# 1.Linear
# 2.Polynomial
# 3.Radial basis function (RBF)
# 4.Sigmoid

# Each of these functions has its characteristics, its pros and cons, and its equation, 
# but as there's no easy way of knowing which function performs best with any given dataset. 
# We usually choose different functions in turn and compare the results. 

# Use the default one, RBF

from sklearn import svm
clf = svm.SVC(kernel='rbf')
clf.fit(X_train, y_train)

# values prediction 
y_hat = clf.predict(X_test)
y_hat

# model evaluation 

from sklearn.metrics import classification_report, confusion_matrix
import itertools

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, y_hat, labels=[2,4])
np.set_printoptions(precision=2)

print (classification_report(y_test, y_hat))

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['Benign(2)','Malignant(4)'],normalize= False,  title='Confusion matrix')

from sklearn.metrics import f1_score, jaccard_score
f1_score(y_test, y_hat, average='weighted') 
jaccard_score(y_test, y_hat, pos_label =2)


## Another model, but using the linear kernel and both accuracy evaluation (jaccard and f1 score)

clf2 = svm.SVC(kernel='linear')
clf2.fit(X_train, y_train) 
y_hat2 = clf2.predict(X_test)

print('The jaccard score is:', jaccard_score(y_test, y_hat2, pos_label=2))
print('The f1 score is:', f1_score(y_test, y_hat2, average='weighted'))







