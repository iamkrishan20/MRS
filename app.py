from flask import Flask, request, jsonify, render_template
from MovieRecomendation import recommend, recommend_by_genre, get_all_movies, get_all_genres

app = Flask(__name__)

# Define the list of genres (can also be dynamic from a database or API)
GENRES = ["Action", "Comedy", "Drama", "Horror", "Romance", "Thriller"]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/recommend', methods=['GET'])
def recommend_movie_or_genre():
    query = request.args.get('query')
    
    print(f"Received query: {query}")  # Debug log

    if not query:
        return jsonify({"error": "A 'query' parameter is required."})

    # Convert query to lowercase for case-insensitive comparison
    query_lower = query.strip().lower()

    # Check if the query is a genre (case-insensitive)
    if query_lower in [genre.lower() for genre in GENRES]:
        # If it's a genre, recommend by genre
        recommended_movies = recommend_by_genre(query)
        print(f"Recommended movies by genre: {recommended_movies}")  # Debug log
    else:
        # Otherwise, treat it as a movie title
        recommended_movies = recommend(movie=query)
        if not recommended_movies:
            recommended_movies = recommend_by_genre(query)  # Fallback if no movies found by title
            print(f"Recommended movies by genre (fallback): {recommended_movies}")  # Debug log

    return jsonify(recommended_movies)

# ✅ Unified search: Fetch all movie titles
@app.route('/get_all_movies', methods=['GET'])
def fetch_all_movies():
    all_movies = get_all_movies()
    print(f"Fetched {len(all_movies)} movies.")  # Debug log
    return jsonify(all_movies)

# ✅ Fetch all genres: Fetch all unique genres
@app.route('/get_all_genres', methods=['GET'])
def fetch_all_genres():
    all_genres = get_all_genres()
    print(f"Fetched {len(all_genres)} genres.")  # Debug log
    return jsonify(all_genres)

if __name__ == '__main__':
    app.run(debug=True)
