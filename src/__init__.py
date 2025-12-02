"""
Movie Analytics Recommender - Source Package

This package contains the core functionality for the movie recommendation system.
"""

from .feature_engineering import (
    # Genre functions
    extract_genres,
    create_genre_encoder,
    encode_genres_onehot,
    # Tag functions
    preprocess_tags,
    create_tfidf_vectorizer,
    vectorize_tags_tfidf,
    # Year functions
    extract_year_from_title,
    extract_years,
    create_decade_feature,
    create_year_features,
    # Pipeline
    FeaturePipeline,
)

from .baseline_model import (
    # Weighted scoring functions
    compute_weighted_popularity_score,
    compute_popularity_scores,
    # Ranking functions
    rank_movies_globally,
    filter_by_minimum_votes,
    # Recommendation functions
    get_top_n_recommendations,
    get_personalized_top_n,
    # Utility functions
    compute_rating_statistics,
    merge_movie_with_stats,
)

__all__ = [
    # Genre functions
    'extract_genres',
    'create_genre_encoder',
    'encode_genres_onehot',
    # Tag functions
    'preprocess_tags',
    'create_tfidf_vectorizer',
    'vectorize_tags_tfidf',
    # Year functions
    'extract_year_from_title',
    'extract_years',
    'create_decade_feature',
    'create_year_features',
    # Pipeline
    'FeaturePipeline',
    # Baseline model - Weighted scoring
    'compute_weighted_popularity_score',
    'compute_popularity_scores',
    # Baseline model - Ranking
    'rank_movies_globally',
    'filter_by_minimum_votes',
    # Baseline model - Recommendations
    'get_top_n_recommendations',
    'get_personalized_top_n',
    # Baseline model - Utilities
    'compute_rating_statistics',
    'merge_movie_with_stats',
]
