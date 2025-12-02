"""
Movie Analytics Recommender - Source Package

This package contains utilities for data inspection and analysis
of the MovieLens dataset for building movie recommendation systems.
"""

from .data_inspection import (
    load_ratings,
    load_movies,
    load_tags,
    inspect_dataframe,
    print_dataset_summary,
    get_ratings_statistics,
    get_movies_statistics,
    get_tags_statistics,
)

__all__ = [
    "load_ratings",
    "load_movies",
    "load_tags",
    "inspect_dataframe",
    "print_dataset_summary",
    "get_ratings_statistics",
    "get_movies_statistics",
    "get_tags_statistics",
]
