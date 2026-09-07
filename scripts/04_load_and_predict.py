import mlflow
from sklearn.datasets import load_breast_cancer


def load_and_predict():
    """
    Simulates a production scenario by loading a model using an alias
    from the MLflow Model Registry and using it for prediction.
    """

    MODEL_NAME = "cancer-classifier-prod"
    MODEL_ALIAS = "staging"

    print(f"Loading model '{MODEL_NAME}' with alias '@{MODEL_ALIAS}'...")

    # Load the model from the MLflow Model Registry using Alias URI
    try:
        model = mlflow.pyfunc.load_model(
            model_uri=f"models:/{MODEL_NAME}@{MODEL_ALIAS}"
        )
    except mlflow.exceptions.MlflowException as e:
        print(f"\nError loading model: {e}")
        print(
            f"Please make sure a model version has the alias "
            f"'@{MODEL_ALIAS}' in the MLflow UI."
        )
        return

    # Load Breast Cancer dataset
    cancer_data = load_breast_cancer(as_frame=True)

    X = cancer_data.data
    y = cancer_data.target
    target_names = cancer_data.target_names

    # Select the first sample of each class
    sample_indices = [
        y[y == 0].index[0],  # First malignant sample
        y[y == 1].index[0]   # First benign sample
    ]

    sample_data = X.loc[sample_indices]
    actual_labels = y.loc[sample_indices]

    # Make predictions using the loaded model
    predictions = model.predict(sample_data)

    print("-" * 60)
    print("Prediction Results")
    print("-" * 60)

    # Display actual/predicted class names instead of 0/1
    for actual, predicted in zip(actual_labels, predictions):
        actual = int(actual)
        predicted = int(predicted)

        actual_name = target_names[actual]
        predicted_name = target_names[predicted]

        correct = actual == predicted

        print(
            f"Actual: {actual_name} | "
            f"Predicted: {predicted_name} | "
            f"Correct: {correct}"
        )

    print("-" * 60)


if __name__ == "__main__":
    load_and_predict()