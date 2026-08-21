"""Dataset adapters and batching utilities."""

from vector_map_former.data.collate import PolygonBatch, make_collate_fn, make_data_loader
from vector_map_former.data.mapgeneralizer import MapGeneralizerDataset

__all__ = [
    "MapGeneralizerDataset",
    "PolygonBatch",
    "make_collate_fn",
    "make_data_loader",
]
