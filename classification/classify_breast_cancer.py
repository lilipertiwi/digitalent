# import package-package yang diperlukan untuk classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import Perceptron
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

# meload dataset iris dan membagi data untuk training dan testing, contoh dibawah data test diambil 25% dan data train 75%
print("[INFO] loading data...")
dataset = load_breast_cancer()
(trainX, testX, trainY, testY) = train_test_split(dataset.data, dataset.target, random_state=3, test_size=0.25)

# data training menggunakan model yang dipilih user
print("[INFO] using '{}' model".format(args["model"]))
model = models[args['model']]
model.fit(trainX,trainY)

# evaluasi data menggunakan setiap baris data test
print("[INFO] evaluating...")
predictions = model.predict(testX)
print(classification_report(testY, predictions, target_names=dataset.target_names))