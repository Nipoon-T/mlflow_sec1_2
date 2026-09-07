import mlflow
from sklearn.datasets import load_breast_cancer


def validate_data():
    """
    Loads the breast cancer dataset, performs basic validation checks,
    and logs the results to MLflow.
    """
    mlflow.set_experiment("Breast Cancer - Data Validation")

    with mlflow.start_run():
        print("Starting data validation run...")
        mlflow.set_tag("ml.step", "data_validation")

        # 1. Load data
        cancer_data = load_breast_cancer(as_frame=True)
        df = cancer_data.frame
        print("Data loaded successfully.")

        # 2. Basic validation
        num_rows, num_cols = df.shape
        num_classes = df["target"].nunique()
        missing_values = df.isnull().sum().sum()

        # Class balance
        class_proportions = df["target"].value_counts(normalize=True)
        class_balance = class_proportions.min()

        print(f"Dataset shape: {num_rows} rows, {num_cols} columns")
        print(f"Number of classes: {num_classes}")
        print(f"Missing values: {missing_values}")
        print(f"Class balance: {class_balance:.4f}")

        # 3. Log results to MLflow
        mlflow.log_metric("num_rows", num_rows)
        mlflow.log_metric("num_cols", num_cols)
        mlflow.log_metric("missing_values", missing_values)
        mlflow.log_metric("class_balance", class_balance)
        mlflow.log_param("num_classes", num_classes)

        # 4. Validation criteria
        validation_status = "Success"

        if (
            missing_values > 0
            or num_classes != 2
            or class_balance < 0.20
        ):
            validation_status = "Failed"

        mlflow.log_param("validation_status", validation_status)
        print(f"Validation status: {validation_status}")

        # 5. Return non-zero exit code when validation fails
        if validation_status == "Failed":
            raise SystemExit(
                "Data validation failed — หยุด pipeline ไม่ให้ไปขั้นถัดไป"
            )

        print("Data validation run finished.")


if __name__ == "__main__":
    validate_data()