import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import FactorAnalysis

iris =datasets.load_iris()

X = iris.data
y = iris.target
target_names = iris.target_names

# PCA
pca = PCA(n_components=2)
X_r = pca.fit(X).transform(X)

# LDA
lda = LinearDiscriminantAnalysis(n_components=2)
X_r2 = lda.fit(X, y).transform(X)

# FA
fa = FactorAnalysis(n_components=2)
X_r3 = fa.fit_transform(iris.data)

print('explained variance ratio (first two components): %s'
     % str(pca.explained_variance_ratio_))

plt.figure()
colors=['navy', 'turquoise', 'darkorange']
lw = 2

for color, i, target_name in zip(colors, [0, 1, 2], target_names):
    plt.scatter(X_r[y == i, 0], X_r[y == i, 1], color=color, alpha=.8, lw=lw,
               label=target_name)
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.title('PCA of IRIS dataset')

plt.figure()
for color, i, target_name in zip(colors, [0, 1, 2], target_names):
    plt.scatter(X_r2[y == i, 0], X_r2[y == i, 1], color=color, alpha=.8,
               label=target_name)
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.title('LDA of IRIS dataset')

plt.figure()
for color, i, target_name in zip(colors, [0, 1, 2], target_names):
    plt.scatter(X_r3[y == i, 0], X_r3[y == i, 1], color=color, alpha=.8,
               label=target_name)
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.title('FA of IRIS dataset')

plt.figure()
for color, i, target_name in zip(colors, [0, 1, 2], target_names):
    plt.scatter(X[y == i, 2], X[y == i, 3], color=color, alpha=.8,
               label=target_name)
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.title('Original of IRIS dataset')

plt.show()