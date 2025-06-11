from typing import Literal
import pandas as pd
import numpy as np
import json
from os.path import join
from pathlib import Path
from pydantic import BaseModel


class Scaler(BaseModel):
    min: float
    max: float


class ColumnPreprocessingInfo(BaseModel):
    values: list[str] | None = None
    scaler: Scaler | None = None


class DataPreprocessor:
    """
    A class for preprocessing pandas DataFrames with common operations.
    """

    def __init__(self, df: pd.DataFrame):
        """
        Initialize the preprocessor with a pandas DataFrame.

        Args:
            df (pd.DataFrame): Input DataFrame to preprocess
        """
        self.df = df.copy()
        self._load_preprocessing_info()
        self._load_final_neighbourhoods()

    def preprocess_airbnb_data(self) -> "DataPreprocessor":
        """
        Apply all preprocessing steps for Airbnb data.

        Returns:
            DataPreprocessor: self for method chaining
        """
        return (
            self.drop_columns(["id", "name", "host_id", "host_name"])
            .process_neighbourhoods()
            .process_review_dates()
            .drop_columns(["reviews_per_month", "days_since_review", "last_review"])
            .process_binary_features()
            .drop_columns(["calculated_host_listings_count", "minimum_nights"])
            .encode_categorical(method="onehot")
            .scale_features()
        )

    def _load_preprocessing_info(self):
        """Load preprocessing info from JSON file."""
        preprocessing_path = join(Path(__file__).parent.parent, "data", "preprocessing_info.json")

        with open(preprocessing_path, "r") as f:
            preprocessing_info = json.load(f)

        self._config_by_col: dict[str, ColumnPreprocessingInfo] = {}
        for col, info in preprocessing_info.items():
            self._config_by_col[col] = ColumnPreprocessingInfo.model_validate(info)

        self._categorical_columns = [
            col
            for col in self._config_by_col
            if self._config_by_col[col].values is not None
        ]
        self._numerical_columns = [
            col
            for col in self._config_by_col
            if self._config_by_col[col].scaler is not None
        ]

    def _load_final_neighbourhoods(self):
        """Load final neighbourhoods from text file."""
        neighbourhoods_path = join(Path(__file__).parent.parent, "data", "final_neighbourhoods.txt")

        with open(neighbourhoods_path, "r") as file:
            self.final_neighbourhoods = [s.strip() for s in file.readlines()]

    def handle_missing_values(
        self,
        strategy: Literal["mean", "median", "mode", "drop"] = "mean",
        columns: list[str] | None = None,
    ) -> "DataPreprocessor":
        """
        Handle missing values in the DataFrame.

        Args:
            - strategy: Strategy to handle missing values ('mean', 'median', 'mode', 'drop')
            - columns: List of columns to apply the strategy. If None, applies to all numeric columns.

        Returns:
            DataPreprocessor: self for method chaining
        """
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns

        if strategy == "drop":
            self.df = self.df.dropna(subset=columns)
        else:
            for col in columns:
                if strategy == "mean":
                    fill_value = self.df[col].mean()
                elif strategy == "median":
                    fill_value = self.df[col].median()
                elif strategy == "mode":
                    fill_value = self.df[col].mode()[0]
                self.df[col] = self.df[col].fillna(fill_value)

        return self

    def remove_duplicates(self) -> "DataPreprocessor":
        """
        Remove duplicate rows from the DataFrame.

        Returns:
            DataPreprocessor: self for method chaining
        """
        self.df = self.df.drop_duplicates()
        return self

    def encode_categorical(
        self, method: Literal["onehot", "label"] = "onehot"
    ) -> "DataPreprocessor":
        """
        Encode categorical variables using fixed categories from preprocessing_info.

        Args:
            - columns: List of categorical columns to encode
            - method: Encoding method ('onehot' or 'label')

        Returns:
            DataPreprocessor: self for method chaining
        """
        for col in self._categorical_columns:
            if method == "onehot":
                # Get categories from preprocessing info
                categories = self._config_by_col[col].values
                # Create dummy columns with fixed categories
                encoded = pd.get_dummies(self.df[col], prefix=col, columns=categories)
                # Ensure all expected columns are present
                for category in categories:
                    dummy_col = f"{col}_{category}"
                    if dummy_col not in encoded.columns:
                        encoded[dummy_col] = 0
                # Convert all encoded columns to integers
                encoded = encoded.astype(np.int8)
                self.df = pd.concat([self.df, encoded[sorted(encoded.columns)]], axis=1)
                self.df.drop(columns=[col], inplace=True)
            elif method == "label":
                self.df[col] = (
                    self.df[col].map(self._config_by_col[col].values).fillna(-1)
                )

        return self

    def scale_features(self) -> "DataPreprocessor":
        """
        Scale numerical features.

        Args:
            - columns: List of columns to scale
            - method: Scaling method ('standard' or 'minmax')

        Returns:
            DataPreprocessor: self for method chaining
        """
        for col in self._numerical_columns:
            min_val = self._config_by_col[col].scaler.min
            max_val = self._config_by_col[col].scaler.max
            self.df[col] = (self.df[col] - min_val) / (max_val - min_val)

        return self

    def drop_columns(self, columns: list[str]) -> "DataPreprocessor":
        """
        Drop specified columns from the DataFrame.

        Args:
            columns: List of columns to drop

        Returns:
            DataPreprocessor: self for method chaining
        """
        self.df = self.df.drop(columns, axis=1)
        return self

    def process_neighbourhoods(self) -> "DataPreprocessor":
        """
        Process neighbourhood data by grouping small districts.

        Returns:
            DataPreprocessor: self for method chaining
        """
        data_in_small_districts = ~self.df["neighbourhood"].isin(
            self.final_neighbourhoods
        )
        self.df.loc[data_in_small_districts, "neighbourhood"] = (
            "small districts in "
            + self.df.loc[data_in_small_districts, "neighbourhood_group"]
        )
        return self

    def process_review_dates(self) -> "DataPreprocessor":
        """
        Process review dates and create review recency categories.

        Returns:
            DataPreprocessor: self for method chaining
        """
        self.df["last_review"] = pd.to_datetime(
            self.df["last_review"], format="%Y-%m-%d", errors="coerce"
        )
        reference_date = self.df["last_review"].max()
        self.df["days_since_review"] = (reference_date - self.df["last_review"]).dt.days
        self.df["review_recency"] = self.df["days_since_review"].apply(
            self._categorize_review_recency
        )
        return self

    def _categorize_review_recency(self, days):
        """Categorize review recency based on days since last review."""
        if pd.isna(days):
            return "No reviews"
        elif days <= 30:
            return "Last month"
        elif days <= 90:
            return "Last quarter"
        elif days <= 365:
            return "Last year"
        else:
            return "Over a year ago"

    def process_binary_features(self) -> "DataPreprocessor":
        """
        Process binary features.

        Returns:
            DataPreprocessor: self for method chaining
        """
        self.df["hosts_multiple_apts"] = (
            self.df["calculated_host_listings_count"] > 1
        ).astype(np.int8)
        self.df["availability_365"] = (self.df["availability_365"] > 0).astype(np.int8)
        return self

    def get_preprocessed_data(self) -> pd.DataFrame:
        """
        Get the preprocessed DataFrame.

        Returns:
            pd.DataFrame: Preprocessed DataFrame
        """
        return self.df.copy()
