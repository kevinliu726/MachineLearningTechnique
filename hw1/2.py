import matplotlib.pyplot as plt
import numpy as np

x = np.array([[1,0],[0,1],[0,-1],[-1,0],[0,2],[0,-2],[-2,0]])
y = np.array([-1,-1,-1,1,1,1,1])

def mykernel(x1,x2):
	tmp = (1 + np.dot(x1,x2.T)) ** 2
	return tmp

from sklearn import svm
clf = svm.SVC(kernel = mykernel, C = 1e10, degree = 2, gamma = 1, coef0 = 1)
clf.fit(x,y)

print('Indices of SV:', clf.support_)
print('SV:', clf.support_vectors_)
print('alpha_n:', y[clf.support_] * clf.dual_coef_[0])

b = clf.intercept_[0]
print(b)
print(x[clf.support_])
print(clf.dual_coef_[0])
