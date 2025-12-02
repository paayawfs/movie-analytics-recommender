"""
Baseline Model Module for Movie Analytics Recommender

This module provides a weighted popularity-based baseline model for movie
recommendations. It uses a combination of average rating and rating count
to compute popularity scores and generate recommendations.
"""

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


# =============================================================================
# Weighted Popularity Scoring
# =============================================================================


def compute_weighted_popularity_score(
    avg_rating: float,
    rating_count: int,
    global_mean: float,
    min_votes: int = 10,
    weight_factor: float = 0.5
) -> float:
    """
    Compute a weighted popularity score for a movie.

    Uses a Bayesian average approach similar to IMDB's weighted rating formula:
    weighted_score = (v / (v + m)) * R + (m / (v + m)) * C

    Where:
    - v = number of votes (ratings) for the movie
    - m = minimum votes required (threshold)
    - R = average rating of the movie
    - C = global mean rating across all movies

    Args:
        avg_rating: Average rating of the movie.
        rating_count: Number of ratings the movie has received.
        global_mean: Global mean rating across all movies.
        min_votes: Minimum number of votes required for consideration.
        weight_factor: Factor to control the influence of min_votes (default: 0.5).

    Returns:
        Weighted popularity score.

    Example:
        >>> score = compute_weighted_popularity_score(
        ...     avg_rating=4.5, rating_count=100, global_mean=3.5, min_votes=10
        ... )
        >>> print(f"{score:.2f}")
        4.41
    """
    # Adjust minimum votes by weight factor
    m = min_votes * weight_factor

    # Bayesian weighted average
    weighted_score = (
        (rating_count / (rating_count + m)) * avg_rating +
        (m / (rating_count + m)) * global_mean
    )

    return weighted_score


def compute_popularity_scores(
    df: pd.DataFrame,
    rating_column: str = "avg_rating",
    count_column: str = "rating_count",
    min_votes: int = 10,
    weight_factor: float = 0.5
) -> pd.Series:
    """
    Compute weighted popularity scores for all movies in a DataFrame.

    Args:
        df: DataFrame containing movie rating statistics.
        rating_column: Column name for average ratings.
        count_column: Column name for rating counts.
        min_votes: Minimum number of votes required for consideration.
        weight_factor: Factor to control the influence of min_votes.

    Returns:
        Series of weighted popularity scores indexed like the input DataFrame.

    Example:
        >>> df = pd.DataFrame({
        ...     'movieId': [1, 2, 3],
        ...     'avg_rating': [4.5, 3.0, 4.0],
        ...     'rating_count': [100, 5, 50]
        ... })
        >>> scores = compute_popularity_scores(df)
    """
    global_mean = df[rating_column].mean()

    scores = df.apply(
        lambda row: compute_weighted_popularity_score(
            avg_rating=row[rating_column],
            rating_count=row[count_column],
            global_mean=global_mean,
            min_votes=min_votes,
            weight_factor=weight_factor
        ),
        axis=1
    )

    return scores


# =============================================================================
# Global Movie Ranking
# =============================================================================


def rank_movies_globally(
    df: pd.DataFrame,
    rating_column: str = "avg_rating",
    count_column: str = "rating_count",
    min_votes: int = 10,
    weight_factor: float = 0.5,
    score_column: str = "popularity_score"
) -> pd.DataFrame:
    """
    Rank all movies globally based on weighted popularity scores.

    Args:
        df: DataFrame containing movie data with rating statistics.
        rating_column: Column name for average ratings.
        count_column: Column name for rating counts.
        min_votes: Minimum number of votes required for consideration.
        weight_factor: Factor to control the influence of min_votes.
        score_column: Name for the new popularity score column.

    Returns:
        DataFrame sorted by popularity score in descending order,
        with an additional column for the computed score.

    Example:
        >>> movies = pd.DataFrame({
        ...     'movieId': [1, 2, 3],
        ...     'title': ['Movie A', 'Movie B', 'Movie C'],
        ...     'avg_rating': [4.5, 3.0, 4.0],
        ...     'rating_count': [100, 5, 50]
        ... })
        >>> ranked = rank_movies_globally(movies)
    """
    # Create a copy to avoid modifying the original
    result_df = df.copy()

    # Compute popularity scores
    result_df[score_column] = compute_popularity_scores(
        df=result_df,
        rating_column=rating_column,
        count_column=count_column,
        min_votes=min_votes,
        weight_factor=weight_factor
    )

    # Sort by popularity score (descending)
    result_df = result_df.sort_values(score_column, ascending=False)

    # Reset index for clean ranking
    result_df = result_df.reset_index(drop=True)

    return result_df


def filter_by_minimum_votes(
    df: pd.DataFrame,
    count_column: str = "rating_count",
    min_votes: int = 10
) -> pd.DataFrame:
    """
    Filter movies that meet the minimum vote threshold.

    Args:
        df: DataFrame containing movie data.
        count_column: Column name for rating counts.
        min_votes: Minimum number of votes required.

    Returns:
        Filtered DataFrame containing only movies with sufficient votes.

    Example:
        >>> df = pd.DataFrame({
        ...     'movieId': [1, 2, 3],
        ...     'rating_count': [100, 5, 50]
        ... })
        >>> filtered = filter_by_minimum_votes(df, min_votes=10)
        >>> len(filtered)
        2
    """
    return df[df[count_column] >= min_votes].copy()


# =============================================================================
# Top-N Recommendations
# =============================================================================


def get_top_n_recommendations(
    df: pd.DataFrame,
    n: int = 10,
    rating_column: str = "avg_rating",
    count_column: str = "rating_count",
    min_votes: int = 10,
    weight_factor: float = 0.5,
    exclude_movie_ids: Optional[List[int]] = None,
    movie_id_column: str = "movieId"
) -> pd.DataFrame:
    """
    Generate top-N movie recommendations based on weighted popularity.

    Args:
        df: DataFrame containing movie data with rating statistics.
        n: Number of recommendations to return.
        rating_column: Column name for average ratings.
        count_column: Column name for rating counts.
        min_votes: Minimum number of votes required for consideration.
        weight_factor: Factor to control the influence of min_votes.
        exclude_movie_ids: List of movie IDs to exclude from recommendations.
        movie_id_column: Column name for movie IDs.

    Returns:
        DataFrame containing top-N recommended movies with popularity scores.

    Example:
        >>> movies = pd.DataFrame({
        ...     'movieId': [1, 2, 3, 4, 5],
        ...     'title': ['Movie A', 'Movie B', 'Movie C', 'Movie D', 'Movie E'],
        ...     'avg_rating': [4.5, 3.0, 4.0, 4.8, 3.5],
        ...     'rating_count': [100, 5, 50, 200, 30]
        ... })
        >>> top_movies = get_top_n_recommendations(movies, n=3)
    """
    # Filter out excluded movies if specified
    filtered_df = df.copy()
    if exclude_movie_ids is not None and len(exclude_movie_ids) > 0:
        filtered_df = filtered_df[
            ~filtered_df[movie_id_column].isin(exclude_movie_ids)
        ]

    # Filter by minimum votes
    filtered_df = filter_by_minimum_votes(
        filtered_df,
        count_column=count_column,
        min_votes=min_votes
    )

    # Rank movies
    ranked_df = rank_movies_globally(
        df=filtered_df,
        rating_column=rating_column,
        count_column=count_column,
        min_votes=min_votes,
        weight_factor=weight_factor
    )

    # Return top-N
    return ranked_df.head(n)


def get_personalized_top_n(
    movie_df: pd.DataFrame,
    user_watched_ids: List[int],
    n: int = 10,
    rating_column: str = "avg_rating",
    count_column: str = "rating_count",
    min_votes: int = 10,
    weight_factor: float = 0.5,
    movie_id_column: str = "movieId"
) -> pd.DataFrame:
    """
    Generate personalized top-N recommendations excluding movies user has watched.

    This is a simple personalization that filters out already-watched movies
    from the global popularity ranking.

    Args:
        movie_df: DataFrame containing movie data with rating statistics.
        user_watched_ids: List of movie IDs the user has already watched.
        n: Number of recommendations to return.
        rating_column: Column name for average ratings.
        count_column: Column name for rating counts.
        min_votes: Minimum number of votes required for consideration.
        weight_factor: Factor to control the influence of min_votes.
        movie_id_column: Column name for movie IDs.

    Returns:
        DataFrame containing top-N recommended movies the user hasn't seen.

    Example:
        >>> movies = pd.DataFrame({
        ...     'movieId': [1, 2, 3, 4, 5],
        ...     'title': ['Movie A', 'Movie B', 'Movie C', 'Movie D', 'Movie E'],
        ...     'avg_rating': [4.5, 3.0, 4.0, 4.8, 3.5],
        ...     'rating_count': [100, 5, 50, 200, 30]
        ... })
        >>> user_watched = [1, 4]  # User has watched movies 1 and 4
        >>> recommendations = get_personalized_top_n(movies, user_watched, n=3)
    """
    return get_top_n_recommendations(
        df=movie_df,
        n=n,
        rating_column=rating_column,
        count_column=count_column,
        min_votes=min_votes,
        weight_factor=weight_factor,
        exclude_movie_ids=user_watched_ids,
        movie_id_column=movie_id_column
    )


# =============================================================================
# Utility Functions
# =============================================================================


def compute_rating_statistics(
    ratings_df: pd.DataFrame,
    movie_id_column: str = "movieId",
    rating_column: str = "rating"
) -> pd.DataFrame:
    """
    Compute rating statistics (average and count) for each movie.

    Args:
        ratings_df: DataFrame containing user ratings.
        movie_id_column: Column name for movie IDs.
        rating_column: Column name for rating values.

    Returns:
        DataFrame with movie ID, average rating, and rating count.

    Example:
        >>> ratings = pd.DataFrame({
        ...     'userId': [1, 1, 2, 2, 3],
        ...     'movieId': [1, 2, 1, 2, 1],
        ...     'rating': [4.0, 3.0, 5.0, 4.0, 4.5]
        ... })
        >>> stats = compute_rating_statistics(ratings)
    """
    stats = ratings_df.groupby(movie_id_column)[rating_column].agg([
        ('avg_rating', 'mean'),
        ('rating_count', 'count')
    ]).reset_index()

    return stats


def merge_movie_with_stats(
    movies_df: pd.DataFrame,
    stats_df: pd.DataFrame,
    movie_id_column: str = "movieId"
) -> pd.DataFrame:
    """
    Merge movie metadata with rating statistics.

    Args:
        movies_df: DataFrame containing movie metadata.
        stats_df: DataFrame containing rating statistics.
        movie_id_column: Column name for movie IDs.

    Returns:
        Merged DataFrame with movie metadata and rating statistics.

    Example:
        >>> movies = pd.DataFrame({
        ...     'movieId': [1, 2, 3],
        ...     'title': ['Movie A', 'Movie B', 'Movie C']
        ... })
        >>> stats = pd.DataFrame({
        ...     'movieId': [1, 2],
        ...     'avg_rating': [4.0, 3.5],
        ...     'rating_count': [100, 50]
        ... })
        >>> merged = merge_movie_with_stats(movies, stats)
    """
    merged = movies_df.merge(
        stats_df,
        on=movie_id_column,
        how='left'
    )

    # Fill NaN values for movies with no ratings
    merged['avg_rating'] = merged['avg_rating'].fillna(0.0)
    merged['rating_count'] = merged['rating_count'].fillna(0).astype(int)

    return merged
