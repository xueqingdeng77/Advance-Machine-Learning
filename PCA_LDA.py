# -*- coding: utf-8 -*-
"""
Created on Sun Feb 19 22:34:00 2017
@author: Xueqing Deng 
@email: xueqingdeng77@gmail.com
@reference: Advanced Machine Learning with Python
"""

# This short code is used to compare the dimensionalities reduction algorithms PCA and LDA

# Dataset: the UCI handwritten digits dataset, distributed as part of scikit-learn
# The dataset is included with 64 variables
import numpy as np 
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import matplotlib.cm as cm

#Import data
digits = load_digits()
data = scale(digits.data)

n_samples,n_features=data.shape
n_digits=len(np.unique(digits.target))
labels=digits.target

#For PCA: n_components can be a number in range [1,n_features]
#For LDA: n_components must be a number in range [1,n_classes-1]
comps_pca=10  # number of components remaining
comps_lda=2

#PCA 
pca=PCA(n_components=comps_pca)
#default: n_components == min(n_samples, n_features)
data_r=pca.fit(data).transform(data)

#Percentage of variance explained by each of the selected components
print('PCA: explained variance ratio (first'+str(comps)+ 'components):' + str(pca.explained_variance_ratio_))
print('PCA: sum of explained variance (first' +str(comps)+ 'components):' +str(sum(pca.explained_variance_ratio_)))

#show the first 2 features in 2D map
x=np.arange(2)
ys=[i+x+(i*x)**2 for i in range(10)]

f1=plt.figure(1)
colors=cm.rainbow(np.linspace(0,1, len(ys)))
for c, i, target_name in zip(colors, [0,2,3,4,5,6,7,8,9], labels):
	plt.scatter(data_r[labels==i,0],data_r[labels==i,1],c=c,alpha=0.5)
	plt.legend()
	plt.title('Scatterplot of Points plotted in first \n'
	'2 PCA Components')

f1.show()

#LDA
lda=LDA(n_components=comps_lda)
data_r2=lda.fit(data,labels).transform(data)
#Percentage of variance explained by each of the selected components
print("LDA: explained variance ratio (first %s components): %s"  % (str(comps), str(lda.explained_variance_ratio_)))
print("LDA: sum of explained variance (first %s components): %s" % (str(comps), str(sum(lda.explained_variance_ratio_))))

#show the first 2 features in 2D map
x=np.arange(2)
ys=[i+x+(i*x)**2 for i in range(10)]

f2=plt.figure(2)
colors=cm.rainbow(np.linspace(0,1, len(ys)))
for c, i, target_name in zip(colors, [0,2,3,4,5,6,7,8,9], labels):
	plt.scatter(data_r2[labels==i,0],data_r2[labels==i,1],c=c,alpha=0.5)
	plt.legend()
	plt.title('Scatterplot of Points plotted in first \n'
	'2 LDA Components')

f2.show()

input("Enter to finish")