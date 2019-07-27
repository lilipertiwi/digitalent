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
# di bawah method akan mengembalikan nilai mean, standar deviasi, min dan max dari masing-masing fitur yang disimpan di features
def extract_color_stats(image):
    (R,G,B) = image.split()
    features = [np.mean(R), np.mean(G), np.mean(B), np.std(R), np.std(G), np.std(B), np.min(R), np.min(G), np.min(B), np.max(R), np.max(G), np.max(B)]
    return features

# mengambil argument lalu diparsing ke dictionary args
# argument -m untuk menggunakan model yang ada, -d untuk menggunakan dataset dari folder 3scenes
# argument -i untuk image/gambar yang akan dievaluasi
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", type=str, default="3scenes",
                help="path to directory containing the '3scenes' dataset")
ap.add_argument("-m", "--model", type=str, default="knn",
                    help="type of python machine learning model to use")
ap.add_argument("-i", "--image", type=str, default="",
                help="path to directory containing the '3scenes' dataset")
args = vars(ap.parse_args())

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
imagePaths = paths.list_images(args["dataset"])
data = []
labels = []

# mengekstrak fitur warna setiap gambar dari setiap dataset 
for imagePath in imagePaths:
    image = Image.open(imagePath)
    features = extract_color_stats(image)
    data.append(features)
    # label menyimpan list kedua dari kanan pada directory gambar
    label = imagePath.split(os.path.sep)[-2]
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

# evaluasi gambar
print("[INFO] hasil prediksi data baru...")
data_individu = []
image = Image.open(args["image"])
features = extract_color_stats(image)
data_individu.append(features)

prediction = model.predict(data_individu)

if prediction == 0:
    print("prediksi: coast")
elif prediction == 1:
    print("prediksi: forest")
elif prediction == 2:
    print("prediksi: house")
else:
    print("prediksi: highway")
    
'''print(classification_report(testY, predictions, target_names=le.classes_))'''