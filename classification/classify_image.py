from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from PIL import Image
from imutils import paths
import numpy as np
import argparse
import os


# method untuk mengekstrak fitur warna RGB pada gambar
# di bawah method akan mengembalikan nilai mean dan standar deviasi dari masing-masing fitur yang disimpan di features
def extract_color_stats(image):
    (R,G,B) = image.split()
    features = [np.mean(R), np.mean(G), np.mean(B), np.std(R), np.std(G), np.std(B)]
    return features

# mengambil argument lalu diparsing ke dictionary args
# argument -m untuk menggunakan model yang ada, -d untuk menggunakan dataset dari folder 3scenes
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", type=str, default="3scenes",
                help="path to directory containing the '3scenes' dataset")
ap.add_argument("-m", "--model", type=str, default="knn",
                    help="type of python machine learning model to use")
args = vars(ap.parse_args())
#print(args) >> {'dataset': 'coast', 'model': 'decision_tree'}

# model-model algoritma machine learning yang dapat digunakan
models = {
    "knn": KNeighborsClassifier(n_neighbors=1),
    "naive_bayes": GaussianNB(),
    "logit": LogisticRegression(solver="lbfgs", multi_class="auto"),
    "svm": SVC(kernel="linear"),
    "decision_tree": DecisionTreeClassifier(),
    "random_forest": RandomForestClassifier(n_estimators=100),
    "mlp": MLPClassifier()
    }

# mengekstrak gambar dari list dataset yang ada
print("[INFO] extracting image features...")
imagePaths = paths.list_images(args["dataset"]) #objek list gambar
data = []
labels = []

# mengekstrak fitur warna setiap gambar dari setiap dataset 
for imagePath in imagePaths:
    image = Image.open(imagePath)
    features = extract_color_stats(image)
    data.append(features)
    # label menyimpan list kedua dari kanan pada directory gambar
    label = imagePath.split(os.path.sep)[-2]
    #print(imagePath) >> 3scenes\coast\coast_arnat59.jpg
    #print(label) >> coast/forest/highway
    labels.append(label)

# konversi label dari string ke integer, misal coast -> 0, forest -> 1, dst  
le = LabelEncoder()
labels = le.fit_transform(labels)

# split data train dan data test 75:25
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25)

# training data menggunakan model yang ada
print("[INFO] using '{}' model".format(args["model"]))
model = models[args["model"]]
model.fit(trainX, trainY)

# evaluasi data
print("[INFO] evaluating...")
predictions = model.predict(testX)
print(classification_report(testY, predictions, target_names=le.classes_))