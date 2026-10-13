
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

DATA_DIR = "data/movielens"

movies = pd.read_csv(f"{DATA_DIR}/movies.csv")
ratings = pd.read_csv(f"{DATA_DIR}/ratings.csv")

# Create a sparse movie-user ratings matrix.
movie_ids = pd.Index(ratings["movieId"].unique())
user_ids = pd.Index(ratings["userId"].unique())

movie_map = {movie_id: i for i, movie_id in enumerate(movie_ids)}
user_map = {user_id: i for i, user_id in enumerate(user_ids)}

row_indices = ratings["movieId"].map(movie_map).to_numpy()
col_indices = ratings["userId"].map(user_map).to_numpy()
rating_values = ratings["rating"].to_numpy(dtype=float)

movie_user_matrix = csr_matrix(
    (rating_values, (row_indices, col_indices)),
    shape=(len(movie_ids), len(user_ids))
)

# Count how many ratings each movie has.
rating_counts = ratings.groupby("movieId")["rating"].count()


def recommend_collaborative(movie_title, top_n=5):
    # Find the selected movie.
    matches = movies[
        movies["title"].str.casefold()
        == movie_title.strip().casefold()
    ]

    if matches.empty:
        return []

    movie_id = matches.iloc[0]["movieId"]

    if movie_id not in movie_map:
        return []

    movie_index = movie_map[movie_id]

    # Compare this movie's ratings pattern with other movies.
    similarities = cosine_similarity(
        movie_user_matrix[movie_index],
        movie_user_matrix
    ).ravel()

    # Rank movies by rating-pattern similarity.
    candidate_ids = []
    for candidate_id, index in movie_map.items():
        if candidate_id == movie_id:
            continue

        # Ignore movies with very little ratings evidence.
        if rating_counts.get(candidate_id, 0) < 3:
            continue

        score = similarities[index]

        if score > 0:
            candidate_ids.append((candidate_id, score))

    candidate_ids.sort(key=lambda item: item[1], reverse=True)

    # Convert the top results to movie titles.
    ranked_ids = [item[0] for item in candidate_ids[:top_n]]
    results = movies[movies["movieId"].isin(ranked_ids)].copy()

    rank_map = {
        movie_id: rank
        for rank, (movie_id, score) in enumerate(candidate_ids[:top_n])
    }

    results["_rank"] = results["movieId"].map(rank_map)
    results = results.sort_values("_rank")

    return results["title"].tolist()


if __name__ == "__main__":
    for test_title in ["Toy Story (1995)", "Spider-Man (2002)"]:
        results = recommend_collaborative(test_title)

        print(f"\nRecommendations for {test_title}:")

        if results:
            for number, title in enumerate(results, start=1):
                print(f"{number}. {title}")
        else:
            print("No recommendations found.")
