"""
Feature Engineering Module for Movie Analytics Recommender

This module provides feature extraction and transformation utilities for the
movie recommendation system, including vectorization methods and feature pipelines.
"""

import re
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer


# =============================================================================
# Genre One-Hot Encoding
# =============================================================================


def extract_genres(genres_str: str, delimiter: str = "|") -> List[str]:
    """
    Extract individual genres from a delimited string.

    Args:
        genres_str: String containing genres separated by delimiter.
        delimiter: Character used to separate genres (default: "|").

    Returns:
        List of individual genre strings.

    Example:
        >>> extract_genres("Action|Comedy|Drama")
        ['Action', 'Comedy', 'Drama']
    """
    if pd.isna(genres_str) or not genres_str:
        return []
    return [genre.strip() for genre in str(genres_str).split(delimiter)]


def create_genre_encoder(genre_lists: List[List[str]]) -> MultiLabelBinarizer:
    """
    Create and fit a MultiLabelBinarizer for genre encoding.

    Args:
        genre_lists: List of genre lists for each movie.

    Returns:
        Fitted MultiLabelBinarizer instance.
    """
    mlb = MultiLabelBinarizer()
    mlb.fit(genre_lists)
    return mlb


def encode_genres_onehot(
    df: pd.DataFrame,
    genre_column: str = "genres",
    delimiter: str = "|"
) -> Tuple[pd.DataFrame, MultiLabelBinarizer]:
    """
    One-hot encode movie genres from a DataFrame column.

    Args:
        df: DataFrame containing movie data.
        genre_column: Name of the column containing genre strings.
        delimiter: Character used to separate genres in the string.

    Returns:
        Tuple of (encoded DataFrame with genre columns, fitted encoder).

    Example:
        >>> df = pd.DataFrame({'genres': ['Action|Comedy', 'Drama|Romance']})
        >>> encoded_df, encoder = encode_genres_onehot(df)
    """
    genre_lists = df[genre_column].apply(
        lambda x: extract_genres(x, delimiter)
    ).tolist()

    mlb = create_genre_encoder(genre_lists)
    genre_encoded = mlb.transform(genre_lists)

    genre_df = pd.DataFrame(
        genre_encoded,
        columns=[f"genre_{g}" for g in mlb.classes_],
        index=df.index
    )

    return genre_df, mlb


# =============================================================================
# Tags TF-IDF Vectorization
# =============================================================================


def preprocess_tags(tags: str) -> str:
    """
    Preprocess tag text for TF-IDF vectorization.

    Args:
        tags: Raw tag string.

    Returns:
        Cleaned and preprocessed tag string.
    """
    if pd.isna(tags) or not tags:
        return ""

    # Convert to lowercase
    text = str(tags).lower()

    # Remove special characters but keep spaces
    text = re.sub(r'[^\w\s]', ' ', text)

    # Normalize whitespace
    text = ' '.join(text.split())

    return text


def create_tfidf_vectorizer(
    max_features: int = 1000,
    min_df: int = 2,
    max_df: float = 0.95,
    ngram_range: Tuple[int, int] = (1, 2)
) -> TfidfVectorizer:
    """
    Create a TF-IDF vectorizer with recommended settings for movie tags.

    Args:
        max_features: Maximum number of features to extract.
        min_df: Minimum document frequency for terms.
        max_df: Maximum document frequency for terms.
        ngram_range: Range of n-grams to consider.

    Returns:
        Configured TfidfVectorizer instance (unfitted).
    """
    return TfidfVectorizer(
        max_features=max_features,
        min_df=min_df,
        max_df=max_df,
        ngram_range=ngram_range,
        stop_words='english'
    )


def vectorize_tags_tfidf(
    df: pd.DataFrame,
    tag_column: str = "tags",
    max_features: int = 1000,
    min_df: int = 2,
    max_df: float = 0.95
) -> Tuple[np.ndarray, TfidfVectorizer]:
    """
    Vectorize movie tags using TF-IDF.

    Args:
        df: DataFrame containing movie data.
        tag_column: Name of the column containing tag strings.
        max_features: Maximum number of TF-IDF features.
        min_df: Minimum document frequency.
        max_df: Maximum document frequency.

    Returns:
        Tuple of (TF-IDF matrix as numpy array, fitted vectorizer).

    Example:
        >>> df = pd.DataFrame({'tags': ['action thriller', 'romantic comedy']})
        >>> tfidf_matrix, vectorizer = vectorize_tags_tfidf(df)
    """
    # Preprocess tags
    processed_tags = df[tag_column].apply(preprocess_tags)

    # Create and fit vectorizer
    vectorizer = create_tfidf_vectorizer(
        max_features=max_features,
        min_df=min_df,
        max_df=max_df
    )
    tfidf_matrix = vectorizer.fit_transform(processed_tags)

    return tfidf_matrix.toarray(), vectorizer


# =============================================================================
# Year Extraction
# =============================================================================


def extract_year_from_title(title: str) -> Optional[int]:
    """
    Extract release year from a movie title.

    Assumes year is in parentheses at the end of the title, e.g., "Toy Story (1995)".

    Args:
        title: Movie title string potentially containing year.

    Returns:
        Extracted year as integer, or None if not found.

    Example:
        >>> extract_year_from_title("Toy Story (1995)")
        1995
        >>> extract_year_from_title("Unknown Movie")
        None
    """
    if pd.isna(title) or not title:
        return None

    # Match year pattern at end of title: (YYYY)
    match = re.search(r'\((\d{4})\)\s*$', str(title))
    if match:
        return int(match.group(1))
    return None


def extract_years(
    df: pd.DataFrame,
    title_column: str = "title"
) -> pd.Series:
    """
    Extract release years from movie titles in a DataFrame.

    Args:
        df: DataFrame containing movie data.
        title_column: Name of the column containing movie titles.

    Returns:
        Series of extracted years (with None for missing values).

    Example:
        >>> df = pd.DataFrame({'title': ['Toy Story (1995)', 'Unknown']})
        >>> years = extract_years(df)
    """
    return df[title_column].apply(extract_year_from_title)


def create_decade_feature(years: pd.Series) -> pd.Series:
    """
    Create decade feature from years.

    Args:
        years: Series of year values.

    Returns:
        Series of decade values (e.g., 1990, 2000, 2010).

    Example:
        >>> years = pd.Series([1995, 2003, 2015])
        >>> decades = create_decade_feature(years)
        >>> list(decades)
        [1990, 2000, 2010]
    """
    # Handle NaN values by using pandas nullable integer operations
    # NaN values will remain NaN in the result
    return years.floordiv(10).mul(10)


def create_year_features(
    df: pd.DataFrame,
    title_column: str = "title"
) -> pd.DataFrame:
    """
    Create year-based features from movie titles.

    Args:
        df: DataFrame containing movie data.
        title_column: Name of the column containing movie titles.

    Returns:
        DataFrame with 'year' and 'decade' columns.

    Example:
        >>> df = pd.DataFrame({'title': ['Toy Story (1995)', 'Avatar (2009)']})
        >>> year_features = create_year_features(df)
    """
    years = extract_years(df, title_column)

    return pd.DataFrame({
        'year': years,
        'decade': create_decade_feature(years)
    }, index=df.index)


# =============================================================================
# Feature Pipeline
# =============================================================================


class FeaturePipeline:
    """
    A pipeline for extracting and combining movie features.

    This class orchestrates the extraction of various features from movie data,
    including genres, tags, and year-based features.

    Attributes:
        genre_encoder: Fitted MultiLabelBinarizer for genres.
        tfidf_vectorizer: Fitted TfidfVectorizer for tags.
        feature_names: List of all feature names.
    """

    def __init__(
        self,
        include_genres: bool = True,
        include_tags: bool = True,
        include_year: bool = True,
        tfidf_max_features: int = 1000
    ):
        """
        Initialize the feature pipeline.

        Args:
            include_genres: Whether to include genre features.
            include_tags: Whether to include tag TF-IDF features.
            include_year: Whether to include year-based features.
            tfidf_max_features: Maximum number of TF-IDF features for tags.
        """
        self.include_genres = include_genres
        self.include_tags = include_tags
        self.include_year = include_year
        self.tfidf_max_features = tfidf_max_features

        self.genre_encoder: Optional[MultiLabelBinarizer] = None
        self.tfidf_vectorizer: Optional[TfidfVectorizer] = None
        self.feature_names: List[str] = []
        self._is_fitted = False

    def fit(
        self,
        df: pd.DataFrame,
        genre_column: str = "genres",
        tag_column: str = "tags",
        title_column: str = "title"
    ) -> "FeaturePipeline":
        """
        Fit the feature pipeline on training data.

        Args:
            df: Training DataFrame.
            genre_column: Column containing genre strings.
            tag_column: Column containing tag strings.
            title_column: Column containing movie titles.

        Returns:
            Self for method chaining.
        """
        self.feature_names = []

        if self.include_genres and genre_column in df.columns:
            genre_lists = df[genre_column].apply(extract_genres).tolist()
            self.genre_encoder = create_genre_encoder(genre_lists)
            self.feature_names.extend(
                [f"genre_{g}" for g in self.genre_encoder.classes_]
            )

        if self.include_tags and tag_column in df.columns:
            processed_tags = df[tag_column].apply(preprocess_tags)
            self.tfidf_vectorizer = create_tfidf_vectorizer(
                max_features=self.tfidf_max_features
            )
            self.tfidf_vectorizer.fit(processed_tags)
            self.feature_names.extend(
                [f"tag_{name}" for name in self.tfidf_vectorizer.get_feature_names_out()]
            )

        if self.include_year and title_column in df.columns:
            self.feature_names.extend(['year', 'decade'])

        self._is_fitted = True
        return self

    def transform(
        self,
        df: pd.DataFrame,
        genre_column: str = "genres",
        tag_column: str = "tags",
        title_column: str = "title"
    ) -> np.ndarray:
        """
        Transform data using the fitted pipeline.

        Args:
            df: DataFrame to transform.
            genre_column: Column containing genre strings.
            tag_column: Column containing tag strings.
            title_column: Column containing movie titles.

        Returns:
            Feature matrix as numpy array.

        Raises:
            ValueError: If the pipeline has not been fitted.
        """
        if not self._is_fitted:
            raise ValueError("Pipeline must be fitted before transform.")

        feature_arrays = []

        if self.include_genres and self.genre_encoder is not None:
            genre_lists = df[genre_column].apply(extract_genres).tolist()
            genre_features = self.genre_encoder.transform(genre_lists)
            feature_arrays.append(genre_features)

        if self.include_tags and self.tfidf_vectorizer is not None:
            processed_tags = df[tag_column].apply(preprocess_tags)
            tag_features = self.tfidf_vectorizer.transform(processed_tags).toarray()
            feature_arrays.append(tag_features)

        if self.include_year and title_column in df.columns:
            year_features = create_year_features(df, title_column).fillna(0).values
            feature_arrays.append(year_features)

        if feature_arrays:
            return np.hstack(feature_arrays)
        return np.array([])

    def fit_transform(
        self,
        df: pd.DataFrame,
        genre_column: str = "genres",
        tag_column: str = "tags",
        title_column: str = "title"
    ) -> np.ndarray:
        """
        Fit and transform in one step.

        Args:
            df: DataFrame to fit and transform.
            genre_column: Column containing genre strings.
            tag_column: Column containing tag strings.
            title_column: Column containing movie titles.

        Returns:
            Feature matrix as numpy array.
        """
        self.fit(df, genre_column, tag_column, title_column)
        return self.transform(df, genre_column, tag_column, title_column)

    def get_feature_names(self) -> List[str]:
        """
        Get the names of all features.

        Returns:
            List of feature names.
        """
        return self.feature_names.copy()
