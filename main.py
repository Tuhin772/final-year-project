import os
from tensorflow.keras.models import load_model

if os.path.exists("model.h5"):
    print("Loading saved model...")
    model = load_model("model.h5")
else:
    print("Training new model...")
    model = build_model(X_train.shape[1])
    train_model(model, X_train)
    model.save("model.h5")