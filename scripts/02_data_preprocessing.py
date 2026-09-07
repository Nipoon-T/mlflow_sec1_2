import os
 
import mlflow
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
 
 
def preprocess_data(test_size=0.25, random_state=42):
    """
    Loads raw data, splits it into training and testing sets,
    and logs the resulting datasets as artifacts in MLflow.
    """
    # Set the experiment name
    mlflow.set_experiment("Breast Cancer - Data Preprocessing") # ตั้งชื่อ experiment สำหรับขั้นตอนนี้
 
    with mlflow.start_run() as run:
        run_id = run.info.run_id
        print(f"Starting data preprocessing run with run_id: {run_id}")
        mlflow.set_tag("ml.step", "data_preprocessing") #บันทึก tag เพื่อระบุขั้นตอนของ pipeline เข้า MLflow UI หน้าเว็บเรา
 
        # 1. Load data as a DataFrame
        cancer_data = load_breast_cancer(as_frame=True)
        df = cancer_data.frame
 
        # 2. Split the data into training and testing sets
        X = df.drop('target', axis=1) # แยก features จาก target variable
        y = df['target'] # แยก target variable
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y) #แยกข้อมูลเป็น train และ test set โดย stratify เพื่อให้สัดส่วนของแต่ละ class ใน target variable เท่าเดิม
 
        # 3. Create a temporary directory to save processed data
        processed_data_dir = "processed_data" #ข้อมูลที่พร้อมเข้าโมเดล
        os.makedirs(processed_data_dir, exist_ok=True)
 
        # Recombine features and target for easy saving
        pd.concat([X_train, y_train], axis=1).to_csv(os.path.join(processed_data_dir, "train.csv"), index=False) #ไฟล์แรก เอา x_train และ y_train มารวมกันแล้วบันทึกเป็น train.csv
        pd.concat([X_test, y_test], axis=1).to_csv(os.path.join(processed_data_dir, "test.csv"), index=False) #ไฟล์ที่สอง เอา x_test และ y_test มารวมกันแล้วบันทึกเป็น test.csv
        print(f"Saved processed data to '{processed_data_dir}' directory.")
 
        # 4. Log parameters and metrics
        mlflow.log_param("test_size", test_size) # บันทึก test_size เป็น parameter ชื่อว่า test_size
        mlflow.log_metric("training_set_rows", len(X_train))
        mlflow.log_metric("test_set_rows", len(X_test))
 
        # 5. Log the processed data directory as an artifact
        mlflow.log_artifacts(processed_data_dir, artifact_path="processed_data") #.log_artifacts คือการเซฟไฟล์หรือโฟลเดอร์ทั้งหมดเป็น เข้า mlflow
        print("Logged processed data as artifacts in MLflow.")
 
        print("-" * 50)
        print("Data preprocessing run finished. Please use the following Run ID for the next step:")
        print(f"Preprocessing Run ID: {run_id}")
        print("-" * 50)
 
if __name__ == "__main__":
    preprocess_data()
