from src.utils import calculate_mse, get_threshold, save_plot


def detect_anomalies(model, X_test):
    reconstructions = model.predict(X_test)

    mse = calculate_mse(X_test, reconstructions)

    threshold = get_threshold(mse)

    anomalies = mse > threshold

    print(f"Total anomalies detected: {sum(anomalies)}")

    save_plot(mse, threshold)