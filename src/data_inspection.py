"""
Data Inspection Module for Movie Analytics Recommender

This module provides utilities for loading and inspecting the MovieLens dataset,
including ratings.csv, movies.csv, and tags.csv files.
"""

import pandas as pd
from pathlib import Path


def load_ratings(data_path: str = "data/ratings.csv") -> pd.DataFrame:
    """
    Load the ratings dataset.

    Args:
        data_path: Path to the ratings.csv file.

    Returns:
        DataFrame containing user ratings with columns:
        userId, movieId, rating, timestamp
    """
    return pd.read_csv(data_path)


def load_movies(data_path: str = "data/movies.csv") -> pd.DataFrame:
    """
    Load the movies dataset.

    Args:
        data_path: Path to the movies.csv file.

    Returns:
        DataFrame containing movie information with columns:
        movieId, title, genres
    """
    return pd.read_csv(data_path)


def load_tags(data_path: str = "data/tags.csv") -> pd.DataFrame:
    """
    Load the tags dataset.

    Args:
        data_path: Path to the tags.csv file.

    Returns:
        DataFrame containing user tags with columns:
        userId, movieId, tag, timestamp
    """
    return pd.read_csv(data_path)


def inspect_dataframe(df: pd.DataFrame, name: str = "DataFrame") -> dict:
    """
    Perform basic inspection of a DataFrame.

    Args:
        df: The DataFrame to inspect.
        name: Name identifier for the dataset.

    Returns:
        Dictionary containing inspection results.
    """
    inspection = {
        "name": name,
        "shape": df.shape,
        "columns": list(df.columns),
        "dtypes": df.dtypes.to_dict(),
        "null_counts": df.isnull().sum().to_dict(),
        "memory_usage_mb": df.memory_usage(deep=True).sum() / (1024 * 1024),
    }
    return inspection


def print_dataset_summary(df: pd.DataFrame, name: str = "DataFrame") -> None:
    """
    Print a formatted summary of the dataset.

    Args:
        df: The DataFrame to summarize.
        name: Name identifier for the dataset.
    """
    print(f"\n{'='*60}")
    print(f"Dataset: {name}")
    print(f"{'='*60}")
    print(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"\nColumn Info:")
    print("-" * 40)
    for col in df.columns:
        null_count = df[col].isnull().sum()
        dtype = df[col].dtype
        print(f"  {col}: {dtype} (nulls: {null_count})")
    print(f"\nMemory Usage: {df.memory_usage(deep=True).sum() / (1024*1024):.2f} MB")
    print(f"\nFirst 5 rows:")
    print(df.head())


def get_ratings_statistics(ratings_df: pd.DataFrame) -> dict:
    """
    Calculate statistics for the ratings dataset.

    Args:
        ratings_df: DataFrame containing ratings data.

    Returns:
        Dictionary with rating statistics.
    """
    stats = {
        "total_ratings": len(ratings_df),
        "unique_users": ratings_df["userId"].nunique(),
        "unique_movies": ratings_df["movieId"].nunique(),
        "rating_mean": ratings_df["rating"].mean(),
        "rating_std": ratings_df["rating"].std(),
        "rating_min": ratings_df["rating"].min(),
        "rating_max": ratings_df["rating"].max(),
        "rating_distribution": ratings_df["rating"].value_counts().sort_index().to_dict(),
    }
    return stats


def get_movies_statistics(movies_df: pd.DataFrame) -> dict:
    """
    Calculate statistics for the movies dataset.

    Args:
        movies_df: DataFrame containing movies data.

    Returns:
        Dictionary with movie statistics.
    """
    # Parse genres (genres are pipe-separated)
    all_genres = movies_df["genres"].str.split("|").explode()
    genre_counts = all_genres.value_counts().to_dict()

    stats = {
        "total_movies": len(movies_df),
        "unique_genres": all_genres.nunique(),
        "genre_distribution": genre_counts,
    }
    return stats


def get_tags_statistics(tags_df: pd.DataFrame) -> dict:
    """
    Calculate statistics for the tags dataset.

    Args:
        tags_df: DataFrame containing tags data.

    Returns:
        Dictionary with tag statistics.
    """
    stats = {
        "total_tags": len(tags_df),
        "unique_users": tags_df["userId"].nunique(),
        "unique_movies": tags_df["movieId"].nunique(),
        "unique_tags": tags_df["tag"].nunique(),
        "top_10_tags": tags_df["tag"].value_counts().head(10).to_dict(),
    }
    return stats


def main():
    """Main function to demonstrate data inspection capabilities."""
    print("Movie Analytics Recommender - Data Inspection")
    print("=" * 60)
    print("\nThis module provides utilities for inspecting the MovieLens dataset.")
    print("\nExpected data files:")
    print("  - data/ratings.csv: User ratings (userId, movieId, rating, timestamp)")
    print("  - data/movies.csv: Movie information (movieId, title, genres)")
    print("  - data/tags.csv: User tags (userId, movieId, tag, timestamp)")
    print("\nUsage:")
    print("  from src.data_inspection import load_ratings, load_movies, load_tags")
    print("  ratings = load_ratings('path/to/ratings.csv')")
    print("  print_dataset_summary(ratings, 'Ratings')")


if __name__ == "__main__":
    main()
