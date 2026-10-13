
import pandas as pd
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# Load the movie dataset
movies = pd.read_csv("data/tmdb_5000_movies.csv")
credits = pd.read_csv("data/tmdb_5000_credits.csv")


# Extract names from genres and keywords
def extract_names(text):
    try:
        items = ast.literal_eval(text)
        return " ".join(item["name"] for item in items)
    except (ValueError, SyntaxError, TypeError):
        return ""


# Extract the top cast members
def extract_cast(text):
    try:
        items = ast.literal_eval(text)
        return " ".join(item["name"] for item in items[:3])
    except (ValueError, SyntaxError, TypeError):
        return ""


# Combine movie data
credits = credits[["title", "cast"]]
movies = movies.merge(credits, on="title", how="left")

movies["genres_text"] = movies["genres"].apply(extract_names)
movies["keywords_text"] = movies["keywords"].apply(extract_names)
movies["cast_text"] = movies["cast"].apply(extract_cast)
movies["overview"] = movies["overview"].fillna("")


movies["combined_features"] = (
    (movies["genres_text"] + " ") * 2 +
    (movies["keywords_text"] + " ") * 3 +
    movies["cast_text"] + " " +
    movies["overview"]
)



# Convert text into numerical vectors
tfidf = TfidfVectorizer(stop_words="english", max_features=10000)
tfidf_matrix = tfidf.fit_transform(movies["combined_features"])


# Calculate similarity between movies
similarity_matrix = cosine_similarity(tfidf_matrix)


# Recommend similar movies

def recommend_movies(movie_title, top_n=5):
    matches = movies[
        movies["title"].str.casefold() == movie_title.strip().casefold()
    ]

    if matches.empty:
        return []

    movie_index = matches.index[0]
    scores = list(enumerate(similarity_matrix[movie_index]))
    scores.sort(key=lambda item: item[1], reverse=True)

    recommendations = []
    seen_titles = {movie_title.strip().casefold()}

    for index, score in scores:
        title = str(movies.iloc[index]["title"]).strip()
        normalized_title = title.casefold()

        if normalized_title not in seen_titles:
            recommendations.append(title)
            seen_titles.add(normalized_title)

        if len(recommendations) >= top_n:
            break

    return recommendations



# Test the recommendation model

if __name__ == "__main__":
    test_movies = ["The Dark Knight Rises", "Titanic", "Iron Man"]

    for title in test_movies:
        print(f"\nRecommendations for {title}:")
        results = recommend_movies(title)

        if results:
            for movie in results:
                print("-", movie)
        else:
            print("Movie not found.")

