# Movie Analytics Recommender

A machine learning-based movie recommendation system built using the MovieLens dataset. This project explores various recommendation algorithms and provides tools for dataset inspection and analysis.

## Project Structure

```
movie-analytics-recommender/
├── data/                       # Dataset directory (not included in repo)
│   ├── ratings.csv             # User ratings
│   ├── movies.csv              # Movie information
│   └── tags.csv                # User-generated tags
├── notebooks/
│   └── ml_exploration.ipynb    # ML exploration and modeling notebook
├── src/
│   └── data_inspection.py      # Data inspection utilities
└── README.md
```

## Dataset

This project uses the [MovieLens dataset](https://grouplens.org/datasets/movielens/), which contains:

### ratings.csv
- **userId**: Unique identifier for each user
- **movieId**: Unique identifier for each movie
- **rating**: User rating (0.5 to 5.0)
- **timestamp**: Time of rating

### movies.csv
- **movieId**: Unique identifier for each movie
- **title**: Movie title with release year
- **genres**: Pipe-separated list of genres

### tags.csv
- **userId**: Unique identifier for each user
- **movieId**: Unique identifier for each movie
- **tag**: User-generated tag
- **timestamp**: Time of tagging

## Setup

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/paayawfs/movie-analytics-recommender.git
cd movie-analytics-recommender
```

2. Install required packages:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

3. Download the MovieLens dataset:
   - Visit [MovieLens](https://grouplens.org/datasets/movielens/)
   - Download the dataset (ml-latest-small or ml-25m)
   - Extract and place `ratings.csv`, `movies.csv`, and `tags.csv` in the `data/` directory

## Usage

### Data Inspection

Use the data inspection module to explore the dataset:

```python
from src.data_inspection import load_ratings, load_movies, load_tags
from src.data_inspection import print_dataset_summary, get_ratings_statistics

# Load datasets
ratings = load_ratings('data/ratings.csv')
movies = load_movies('data/movies.csv')
tags = load_tags('data/tags.csv')

# Inspect datasets
print_dataset_summary(ratings, 'Ratings')
print_dataset_summary(movies, 'Movies')
print_dataset_summary(tags, 'Tags')

# Get statistics
stats = get_ratings_statistics(ratings)
print(stats)
```

### ML Exploration Notebook

Run the Jupyter notebook for interactive exploration:

```bash
jupyter notebook notebooks/ml_exploration.ipynb
```

The notebook includes:
- Dataset loading and inspection
- Exploratory Data Analysis (EDA)
- Data preprocessing
- Initial modeling setup with baseline predictors

## Features

### Data Inspection (`src/data_inspection.py`)

- `load_ratings()`: Load ratings.csv
- `load_movies()`: Load movies.csv
- `load_tags()`: Load tags.csv
- `inspect_dataframe()`: Get DataFrame inspection results
- `print_dataset_summary()`: Print formatted dataset summary
- `get_ratings_statistics()`: Calculate rating statistics
- `get_movies_statistics()`: Calculate movie statistics
- `get_tags_statistics()`: Calculate tag statistics

### ML Exploration Notebook

- Rating distribution analysis
- Genre distribution visualization
- User activity analysis
- Movie popularity analysis
- User-item matrix creation
- Baseline model evaluation

## Next Steps

- Implement collaborative filtering algorithms
- Add matrix factorization (SVD, ALS)
- Explore deep learning approaches
- Build hybrid recommendation models
- Implement comprehensive evaluation metrics

## License

This project is open source and available under the MIT License