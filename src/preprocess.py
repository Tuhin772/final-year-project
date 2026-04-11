import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def load_data():
    data = pd.read_csv(
        "data/household_power_consumption.csv",
        sep=',',
        low_memory=False,
        na_values=['?']
    )
    return data


def preprocess_data(data):
    # Use only one column (simple & effective)
    data = data[['Global_active_power']]

    # Remove missing values
    data = data.dropna()

    # Convert to float
    data['Global_active_power'] = data['Global_active_power'].astype(float)

    # Normalize
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)

    # Train-test split
    train_size = int(0.8 * len(data_scaled))
    X_train = data_scaled[:train_size]
    X_test = data_scaled[train_size:]

    return X_train, X_test