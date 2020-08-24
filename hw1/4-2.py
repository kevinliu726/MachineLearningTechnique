import matplotlib.pyplot as plt
import numpy as np

x = np.array([[1,0],[0,1],[0,-1],[-1,0],[0,2],[0,-2],[-2,0]])
y = np.array([-1,-1,-1,1,1,1,1])

def my_decision_function(xy):
    l = []
    for i in xy:
        x1, x2 = i[0], i[1]
        tmp = (x2**2) - 2*x1 - 3/2
        l.append(tmp)
    return np.array(l)

def mykernel(x1, x2):
    return (1 + np.dot(x1, x2.T))**2

from sklearn import svm
clf = svm.SVC(kernel = mykernel, C = 1e10)
clf.fit(x,y)

for i, j in enumerate(x):
    t1, t2 = j[0], j[1]
    coordinate = "x%d(%s,%s)\n" % (i, t1, t2)
    if y[i] == 1:
        plt.scatter(t1, t2, c='r', edgecolors='k')
        plt.annotate(coordinate, (t1,t2),  fontsize=10)
    elif y[i] == -1:
        plt.scatter(t1, t2, c='b', edgecolors='k')
        plt.annotate(coordinate, (t1,t2),  fontsize=10)


ax = plt.gca()
xlim = (-3,3)
ylim = (-3,3)

# create grid to evaluate model
xx = np.linspace(xlim[0], xlim[1], 30)
yy = np.linspace(ylim[0], ylim[1], 30)
YY, XX = np.meshgrid(yy, xx)
xy = np.vstack([XX.ravel(), YY.ravel()]).T
Z = my_decision_function(xy).reshape(XX.shape)

# plot decision boundary and margins
ax.contour(XX, YY, Z, colors='g', levels=[0], alpha=1,linestyles=['-'])

#plt.xlim(-2.0,2.0)
plt.savefig('4-2.png')
plt.show()
