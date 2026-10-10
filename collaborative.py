
import pandas as pd


DATA_DIR = "data/movielens"

movies = pd.read_csv(f"{DATA_DIR}/movies.csv")
ratings = pd.read_csv(f"{DATA_DIR}/ratings.csv")


def recommend_collaborative(movie_title, top_n=5):
    # Find the selected movie.
    matches = movies[
        movies["title"].str.casefold()
        == movie_title.strip().casefold()
    ]

    if matches.empty:
        return []

    movie_id = matches.iloc[0]["movieId"]

    # Find users who rated this movie 4 stars or higher.
    liked_users = ratings[
        (ratings["movieId"] == movie_id)
        & (ratings["rating"] >= 4.0)
    ]["userId"].unique()

    if len(liked_users) == 0:
        return []

    # Find other movies rated by those users.
    other_ratings = ratings[
        ratings["userId"].isin(liked_users)
        & (ratings["movieId"] != movie_id)
    ]

    # Rank movies by their average rating and number of ratings.
    summary = (
        other_ratings.groupby("movieId")["rating"]
        .agg(["mean", "count"])
        .reset_index()
    )

    # Avoid recommendations based on only one rating.
    summary = summary[summary["count"] >= 2]

    summary = summary.sort_values(
        ["mean", "count"],
        ascending=False
    ).head(top_n)

    results = summary.merge(movies, on="movieId", how="left")

    return results["title"].tolist()


if __name__ == "__main__":
    test_title = "Toy Story (1995)"
    results = recommend_collaborative(test_title)

    print(f"Collaborative recommendations for {test_title}:")

    for movie in results:
        print("-", movie)

    if not results:
        print("No recommendations found for this movie.")