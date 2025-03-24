import numpy as np
import pandas as pd
movies = pd.read_csv('tmdb_5000_movies.csv')
credits = pd.read_csv('tmdb_5000_credits.csv')
movies.head()
credits.head()
movies = movies.merge(credits , on = 'title')
movies.head()
movies = movies[['movie_id','title','overview','genres','keywords','cast','crew']]
movies.info()
movies.head()
movies.isnull().sum()
movies.dropna(inplace=True)
movies.isnull().sum()
movies.duplicated().sum()
movies.iloc[0].genres
import ast
def convert(obj):
    L = []
    for i in ast.literal_eval (obj):
        i['name']
        L.append(i['name'])
    return L


movies['genres'] = movies['genres'].apply(convert)
movies.head()
movies['keywords'] = movies['keywords'].apply(convert)
movies.head()
def convert3(obj):
    L = []
    counter = 0
    for i in ast.literal_eval (obj):
        if counter != 3:
           L.append(i['name'])
           counter+=1
        else:
            break
    return L


movies['cast'] = movies['cast'].apply(convert3)
movies.head()
def fetch_director(obj):
    L = []
    counter = 0
    for i in ast.literal_eval (obj):
        if i['job'] == 'Director':
           L.append(i['name'])
           break
    return L
movies['crew'] = movies['crew'].apply(fetch_director)
movies.head()
movies['overview'][0]    # this is string covert into list so we can cancatinate with other
movies['overview'] = movies['overview'].apply(lambda x:x.split())
movies.head()
movies['genres'] = movies['genres'].apply(lambda x:[i.replace(" ","")for i in x])
movies['keywords'] = movies['keywords'].apply(lambda x:[i.replace(" ","")for i in x])
movies['cast'] = movies['cast'].apply(lambda x:[i.replace(" ","")for i in x])
movies['crew'] = movies['crew'].apply(lambda x:[i.replace(" ","")for i in x])
movies.head()
movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']
movies.head()
new_df  = movies[['movie_id','title','tags']]
new_df.loc[:, 'tags'] = new_df['tags'].apply(lambda x: " ".join(x))
new_df.head()
new_df['tags'][0]
new_df.head()
import nltk
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
def stem(text):
    y = []
    for i in text.split():
       y.append(ps.stem(i))
    return " ".join(y)   
new_df['tags'] = new_df['tags'].apply(lambda x: x.lower()).apply(stem) 

new_df['tags'][0]
from sklearn.feature_extraction.text import CountVectorizer
cv  = CountVectorizer(max_features=5000,stop_words='english')
vectors = cv.fit_transform(new_df['tags']).toarray()
vectors
vectors[0]
cv.get_feature_names_out()
ps.stem('loving')
stem('in the 22nd century, a paraplegic marine is dispatched to the moon pandora on a unique mission, but becomes torn between following orders and protecting an alien civilization. action adventure fantasy sciencefiction cultureclash future spacewar spacecolony society spacetravel futuristic romance space alien tribe alienplanet cgi marine soldier battle loveaffair antiwar powerrelations mindandsoul 3d samworthington zoesaldana sigourneyweaver jamescameron')
from sklearn.metrics.pairwise import cosine_similarity
similarity = cosine_similarity(vectors)
sorted(list(enumerate(similarity[0])),reverse=True,key=lambda x:x[1])[1:6]
# Create the fuzzy search function for movie suggestions
from fuzzywuzzy import process
def get_movie_suggestions(query):
    query = query.lower()  # Case-insensitive search
    movie_choices = new_df['title'].tolist()
    closest_matches = process.extract(query, movie_choices, limit=5)  # Get top 5 closest matches
    
    if closest_matches[0][1] < 80:  # If no close match is found with a reasonable score
        return "No close match found in the dataset."
    
    return [match[0] for match in closest_matches]  # Return the movie titles of the closest matches

def recommend(movie=None, genre=None):
    if movie:
        movie = movie.lower()  # Convert the movie title to lowercase
        movie_index = new_df[new_df['title'].str.lower() == movie]  # Find the movie index
        
        if movie_index.empty:  # If no movie is found
            # If movie is not found, suggest similar titles using fuzzy matching
            return ["No movie found with that title. Did you mean one of these?", get_movie_suggestions(movie)]
        
        movie_index = movie_index.index[0]  # Get the index of the movie
        distances = similarity[movie_index]  # Get the similarity scores for the movie
        movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]  # Get top 5 similar movies

        recommended_movies = [new_df.loc[i[0], 'title'] for i in movies_list]  # Get the movie titles
        return recommended_movies
    
    elif genre:  # If genre is provided
        genre = genre.lower()  # Convert genre to lowercase
        from MovieRecomendation import recommend_by_genre  # Import inside the function to avoid circular import
        return recommend_by_genre(genre)  # Call recommend_by_genre with the genre
    
    else:
        return []  # Return an empty list if neither movie nor genre is provided



from fuzzywuzzy import process
# Create a fuzzy search function for genre suggestions
def get_genre_suggestions(query):
    query = query.lower()  # Convert to lowercase to handle case-insensitive search
    
    # List all unique genres in the dataset
    genre_choices = list(set([genre for genres in movies['genres'] for genre in genres]))  
    
    # Find the closest matching genres based on the user's input
    closest_matches = process.extract(query, genre_choices, limit=3)  # Get top 3 closest matches

    # If no match with reasonable confidence, return a message
    if closest_matches[0][1] < 80:  
        return None  # No close match found
    
    # Return the genre with the highest confidence (first match)
    return closest_matches[0][0]


def recommend_by_genre(genre):
    genre = genre.lower()  # Convert genre input to lowercase for case-insensitivity

    # Try to find an exact genre match
    genre_movies = movies[movies['genres'].apply(lambda x: genre in [g.lower() for g in x])]  
    
    if genre_movies.empty:  # If no exact match is found, use fuzzy matching
        matched_genre = get_genre_suggestions(genre)  # Use fuzzy matching for genres
        
        if matched_genre:  # If a close genre match is found, use it
            print(f"Did you mean: {matched_genre}?")  # Optional: show the user the matched genre
            genre_movies = movies[movies['genres'].apply(lambda x: matched_genre.lower() in [g.lower() for g in x])]
        else:
            return ["No movies found for that genre."]  # If no genre match is found at all

    # Continue with the movie recommendations for the matched genre
    genre_movies_ids = genre_movies['movie_id'].tolist()
    genre_movies_df = new_df[new_df['movie_id'].isin(genre_movies_ids)]
    genre_vectors = cv.transform(genre_movies_df['tags']).toarray()
    genre_similarity = cosine_similarity(genre_vectors)
    genre_similarity_sum = genre_similarity.sum(axis=0) / genre_similarity.shape[0]
    movies_list = sorted(list(enumerate(genre_similarity_sum)), reverse=True, key=lambda x: x[1])[1:6]
    
    recommended_movies = [genre_movies_df.iloc[i[0]]['title'] for i in movies_list]
    return recommended_movies



print(recommend('spider-man'))
print(recommend_by_genre('drama'))


def get_all_movies():
    """Returns a list of all movie titles in the dataset."""
    return new_df['title'].tolist()

# Example usage
print(get_all_movies())

# Function to get all unique genres
def get_all_genres():
    """Returns a list of all unique genres in the dataset."""
    # Flatten the list of genres in the 'genres' column and get unique genres
    all_genres = [genre for genres in movies['genres'] for genre in genres]
    unique_genres = list(set(all_genres))  # Remove duplicates
    return unique_genres

# Example usage of the function:
all_genres = get_all_genres()
print(all_genres)  # This will print all unique genres
