# import package-package yang diperlukan untuk classification
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.datasets import load_iris
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import Perceptron
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import FactorAnalysis
import argparse

# mengambil argument setelah run -i <nama file.py>
ap = argparse.ArgumentParser()
# argument yang digunakan untuk menentukan model algoritma klasifikasi yang ingin digunakan
ap.add_argument("-m", "--model", type=str, default="knn",
                    help="type of python machine learning model to use")
args = vars(ap.parse_args())

# dictionary dari model-model yang dapat digunakan
models = {
    "knn": KNeighborsClassifier(n_neighbors=1),
    "naive_bayes": GaussianNB(),
    "logit": LogisticRegression(solver="lbfgs", multi_class="auto"),
    "svm": SVC(kernel="rbf", gamma="auto"),
    "decision_tree": DecisionTreeClassifier(),
    "pct": Perceptron(),
    "random_forest": RandomForestClassifier(n_estimators=100),
    "mlp": MLPClassifier()
    }

iris =datasets.load_iris()
#iris =datasets.load_breast_cancer()
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

print("[INFO] Dataset Original 4 Fitur...")
(trainX, testX, trainY, testY) = train_test_split(X, y, random_state=3, test_size=0.25)
model = models[args['model']]
model.fit(trainX,trainY)
predictions = model.predict(testX)
print(classification_report(testY, predictions, target_names=target_names))

print("[INFO] Dataset Original 2 Fitur...")
(trainX, testX, trainY, testY) = train_test_split(X[:,0:2], y, random_state=3, test_size=0.25)
model = models[args['model']]
model.fit(trainX,trainY)
predictions = model.predict(testX)
print(classification_report(testY, predictions, target_names=target_names))

# PCA
print("[INFO] Dataset PCA 2 Fitur...")
(trainX, testX, trainY, testY) = train_test_split(X_r, y, random_state=3, test_size=0.25)
model = models[args['model']]
model.fit(trainX,trainY)
predictions = model.predict(testX)
print(classification_report(testY, predictions, target_names=target_names))

# LDA
print("[INFO] Dataset LDA 2 Fitur...")
(trainX, testX, trainY, testY) = train_test_split(X_r2, y, random_state=3, test_size=0.25)
model = models[args['model']]
model.fit(trainX,trainY)
predictions = model.predict(testX)
print(classification_report(testY, predictions, target_names=target_names))

# FA
print("[INFO] Dataset FA 2 Fitur...")
(trainX, testX, trainY, testY) = train_test_split(X_r3, y, random_state=3, test_size=0.25)
model = models[args['model']]
model.fit(trainX,trainY)
predictions = model.predict(testX)
print(classification_report(testY, predictions, target_names=target_names))