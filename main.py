from src.preprocess import load_data, preprocess_data
from src.model import build_model, train_model
from src.detect import detect_anomalies

import os
from tensorflow.keras.models import load_model


print("Loading data...")
data = load_data()

print("Preprocessing...")
X_train, X_test = preprocess_data(data)

# Check if model already exists
if os.path.exists("model.h5"):
    print("Loading saved model...")
    model = load_model("model.h5")
else:
    print("Building model...")
    model = build_model(X_train.shape[1])

    print("Training model...")
    train_model(model, X_train)

print("Detecting anomalies...")
detect_anomalies(model, X_test)