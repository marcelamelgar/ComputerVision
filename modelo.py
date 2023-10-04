## marce melgar
## 20200487

# Libraries used
import cv2 as cv
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from joblib import dump

def Modelo():
    searcher = os.listdir('ModeloPlacas')
    X = []
    y = []

    for i in range(len(searcher)):
        photos = os.listdir(f'ModeloPlacas/{searcher[i]}')
        for j in range(len(photos)):
            letter = cv.imread(f'ModeloPlacas/{searcher[i]}/{photos[j]}', cv.IMREAD_GRAYSCALE)
            letter_reduced = letter.flatten()
            X.append(letter_reduced)
            y.append(searcher[i])

    # Train - test datasets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=100, stratify=y)

    # Using random forest at first
    model = RandomForestClassifier(n_estimators = 100, max_features = 45)
    model.fit(X_train, y_train)

    # Predicting values and checking accuracy
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))

    # Making persistent the model
    dump_model = dump(model, "model.sav") 

    return dump_model

Modelo()