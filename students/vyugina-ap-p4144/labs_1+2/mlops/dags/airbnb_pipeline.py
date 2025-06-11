from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import structlog

from airflow import DAG
from airflow.exceptions import AirflowSkipException
from airflow.operators.python import PythonOperator

from mlops.operators.preprocessor import DataPreprocessor
from mlops.operators.qdrant_loader import QdrantLoader
from mlops.settings import settings


logger = structlog.get_logger(__name__)
loader = QdrantLoader()

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "start_date": datetime(2024, 1, 1),
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


def check_new_data(**_) -> list[str]:
    """Check if there are new data files to process"""
    data_dir = Path(settings.data_dir)
    new_files = sorted(data_dir.glob("*.csv"))  # Sort files for consistent order
    if not new_files:
        raise AirflowSkipException("No new data files found")
    return [str(f) for f in new_files]


def process_data(**context) -> list[pd.DataFrame]:
    """Process the data using our preprocessor"""
    input_files = context["task_instance"].xcom_pull(task_ids="check_new_data")
    processed_dataframes = []

    for file_path in input_files:
        try:
            data = pd.read_csv(file_path)
            preprocessor = DataPreprocessor(data)
            processed_data = (
                preprocessor.preprocess_airbnb_data().get_preprocessed_data()
            )
            processed_dataframes.append(processed_data)
        except Exception as e:
            logger.error(f"Error processing {file_path}: {str(e)}", exc_info=True)
            continue

    if not processed_dataframes:
        raise AirflowSkipException("No data was successfully processed")

    return processed_dataframes


def save_to_db(**context):
    """Save processed data to db"""
    processed_dataframes = context["task_instance"].xcom_pull(task_ids="process_data")

    for df in processed_dataframes:
        loader.save_data(df)

with DAG(
    "airbnb_pipeline",
    default_args=default_args,
    description="Airbnb data processing pipeline",
    schedule_interval=timedelta(days=1),
    catchup=False,
    tags=["airbnb", "mlops"],
) as dag:
    (
        PythonOperator(
            task_id="check_new_data",
            python_callable=check_new_data,
        )
        >> PythonOperator(
            task_id="process_data",
            python_callable=process_data,
        )
        >> PythonOperator(
            task_id="save_to_qdrant",
            python_callable=save_to_db,
        )
    )
