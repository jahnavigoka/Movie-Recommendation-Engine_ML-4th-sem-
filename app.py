
import streamlit as st

st.set_page_config(
    page_title="CineMatch",
    page_icon="🎬",
    layout="wide"
)

# Temporary sample results for Day 1.
# We will connect the real ML model on Day 2.
sample_movies = {
    "Avatar": [
        "Guardians of the Galaxy",
        "John Carter",
        "Star Trek",
        "The Avengers",
        "Aliens"
    ],
    "Titanic": [
        "Romeo + Juliet",
        "The Great Gatsby",
        "Atonement",
        "Pearl Harbor",
        "The Notebook"
    ],
    "Iron Man": [
        "Iron Man 2",
        "The Avengers",
        "Captain America: The First Avenger",
        "Thor",
        "Ant-Man"
    ],
    "The Dark Knight Rises": [
        "The Dark Knight",
        "Batman Begins",
        "Man of Steel",
        "Batman",
        "Watchmen"
    ]
}

st.title("🎬 CineMatch")
st.subheader("Discover your next favourite movie")

st.write(
    "Explore movie recommendations based on movie content "
    "and, later, user rating preferences."
)

st.info(
    "Day 1 preview: sample recommendations are displayed "
    "until the machine learning model is connected."
)

method = st.radio(
    "Choose recommendation method",
    ["Content-Based", "Collaborative Filtering"],
    horizontal=True
)

movie_title = st.selectbox(
    "Choose a movie",
    list(sample_movies.keys())
)

if st.button("Recommend Movies", type="primary"):
    st.subheader(f"Movies recommended for: {movie_title}")

    if method == "Content-Based":
        recommendations = sample_movies[movie_title]
    else:
        recommendations = [
            "The Avengers",
            "Inception",
            "Interstellar",
            "The Prestige",
            "The Matrix"
        ]

    columns = st.columns(5)

    for i, movie in enumerate(recommendations):
        with columns[i]:
            st.markdown(f"### {i + 1}")
            st.write(movie)

st.caption(
    "CineMatch — Student Machine Learning Project"
)