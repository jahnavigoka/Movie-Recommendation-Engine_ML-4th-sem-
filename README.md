# MovieVerse — Movie Recommendation Engine

## Project Overview
MovieVerse is a movie recommendation system that suggests movies based on movie content and user rating preferences.

## Features
- Content-Based Filtering using TF-IDF and cosine similarity.
- Collaborative Filtering using the MovieLens ratings dataset.
- Interactive web interface built with Streamlit.
- Movie selection and display of five recommendations.

## Technologies Used
- Python
- Streamlit
- Pandas
- NumPy
- Scikit-learn

## Datasets
- **TMDB 5000 Movies Dataset:** Used for movie genres, keywords, cast, and plot descriptions.
- **MovieLens Latest Small Dataset:** Used for movie information and user ratings.

Place the files in these locations:
- `data/tmdb_5000_movies.csv`
- `data/tmdb_5000_credits.csv`
- `data/movielens/movies.csv`
- `data/movielens/ratings.csv`

## How to Run

1. Install Python.
2. Open the project folder in VS Code.
3. Create and activate a virtual environment.
4. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

5. Start the application:

   ```bash
   python -m streamlit run app.py
   ```

6. Open the local URL displayed in the terminal.

## Project Structure

```text
CineMatch/
├── app.py
├── recommender.py
├── collaborative.py
├── requirements.txt
├── README.md
└── data/
    ├── tmdb_5000_movies.csv
    ├── tmdb_5000_credits.csv
    └── movielens/
        ├── movies.csv
        └── ratings.csv
```

## Team
- Yasasri Pabbisetty
- Jahnavi

## Note
This is an academic machine learning project. Recommendation quality depends on the available dataset and the implemented algorithms.
