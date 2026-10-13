import streamlit as st
from recommender import movies as content_movies, recommend_movies
from collaborative import movies as collaborative_movies, recommend_collaborative

st.set_page_config(
    page_title="CineMatch",
    page_icon="🎬",
    layout="wide"
)

st.title("🎬 CineMatch")
st.subheader("Discover your next favourite movie")

st.write(
    "Get movie recommendations using movie content "
    "or user rating preferences."
)

method = st.radio(
    "Choose recommendation method",
    ["Content-Based", "Collaborative Filtering"],
    horizontal=True
)

if method == "Content-Based":
    movie_titles = sorted(content_movies["title"].dropna().unique())
else:
    movie_titles = sorted(collaborative_movies["title"].dropna().unique())

movie_title = st.selectbox("Choose a movie", movie_titles)

if st.button("Recommend Movies", type="primary"):
    with st.spinner("Finding recommendations..."):
        if method == "Content-Based":
            recommendations = recommend_movies(movie_title)
        else:
            recommendations = recommend_collaborative(movie_title)

    st.subheader(f"Movies recommended for: {movie_title}")

    if recommendations:
        for i, movie in enumerate(recommendations, start=1):
            st.write(f"**{i}.** {movie}")
    else:
        st.warning("No recommendations found for this movie.")

st.caption("CineMatch — Student Machine Learning Project")
