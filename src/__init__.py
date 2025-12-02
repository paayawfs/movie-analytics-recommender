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
]
