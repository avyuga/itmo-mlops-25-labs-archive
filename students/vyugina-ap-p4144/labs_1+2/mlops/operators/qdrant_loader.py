from qdrant_client import QdrantClient  # type: ignore
from qdrant_client.models import Record, VectorParams, Distance
import pandas as pd

from mlops.settings import settings


class QdrantLoader:
    def __init__(self):
        self._qdrant = QdrantClient(settings.qdrant_url)
        if not self._qdrant.collection_exists(settings.qdrant_collection_name):
            self._qdrant.create_collection(
                collection_name=settings.qdrant_collection_name,
                vectors_config=VectorParams(size=settings.qdrant_dimension, distance=Distance.COSINE),
            )

    def save_data(self, data: pd.DataFrame):
        vectors = list(data.to_numpy())
        ids = list(data.index)

        self._qdrant.upload_points(
            collection_name=settings.qdrant_collection_name,
            points=[Record(id=id, vector=vec) for id, vec in zip(ids, vectors)],
            batch_size=settings.qdrant_batch_size,
        )
