from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense


def build_model(input_dim):
    input_layer = Input(shape=(input_dim,))

    encoded = Dense(8, activation='relu')(input_layer)
    encoded = Dense(4, activation='relu')(encoded)

    decoded = Dense(8, activation='relu')(encoded)
    decoded = Dense(input_dim, activation='sigmoid')(decoded)

    autoencoder = Model(inputs=input_layer, outputs=decoded)

    autoencoder.compile(optimizer='adam', loss='mse')

    return autoencoder


def train_model(model, X_train):
    model.fit(
        X_train, X_train,
        epochs=30,
        batch_size=32,
        validation_split=0.1,
        verbose=1
    )

    model.save("model.h5")