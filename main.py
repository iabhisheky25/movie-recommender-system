import numpy as np
import pandas as pd
movies = pd.read_csv('tmdb_5000_movies.csv')
credits = pd.read_csv('tmdb_5000_credits.csv')
movies.head(1)
credits.head(1)
movies = movies.merge(credits,on='title')
movies.head(1)
#genre
#id
#keywords
#title
#overview
#caste
#crew
movies = movies[['movie_id','title','overview','genres','keywords','cast','crew']]
movies.head()
movies.isnull().sum()
movies.dropna(inplace=True)
movies.isnull().sum()
movies.duplicated().sum()
movies.iloc[0].genres
#[{"id": 28, "name": "Action"}, {"id": 12, "name": "Adventure"}, {"id": 14, "name": "Fantasy"}, {"id": 878, "name": "Science Fiction"}] = ['Action','Adventure',fantasy','SciFi]
import ast
def convert (obj):
    L = []
    for i in ast.literal_eval(obj):
        L.append(i['name'])
    return L
movies['genres'] = movies['genres'].apply(convert) 
movies.head()
movies['keywords'] = movies['keywords'].apply(convert)
movies.head() 
movies['cast'] = movies['cast'].apply(convert)
movies.head() 
movies['crew'] = movies['crew'].apply(convert)
movies.head()   
movies['overview'] = movies['overview'].apply(lambda x:x.split())
movies.head()
movies['genres'] = movies['genres'].apply(lambda x:[i.replace(" ","")for i in x])
movies['keywords'] = movies['keywords'].apply(lambda x:[i.replace(" ","")for i in x])
movies['cast'] = movies['cast'].apply(lambda x:[i.replace(" ","")for i in x])
movies['crew'] = movies['crew'].apply(lambda x:[i.replace(" ","")for i in x])
movies.head()
movies['tags'] = movies['overview']+movies['keywords']+movies['cast']+movies['crew']
movies.head()
new_def = movies[['movie_id','title','tags']]
new_def
new_def['tags'] = new_def['tags'].apply(lambda x:" ".join(x))
new_def.head()
new_def['tags'][0]
new_def['tags'] = new_def['tags'].apply(lambda x:x.lower())
new_def.head()
new_def['tags'][0]
new_def['tags'][1]
import sklearn
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=5000,stop_words='english')
vectors = cv.fit_transform(new_def['tags']).toarray()
vectors
vectors[0]
cv.get_feature_names_out()
import nltk
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
def stem(text):
    y = []
    for i in text.split():
        y.append(ps.stem(i))
    return " ".join(y)
new_def['tags'] = new_def['tags'].apply(stem)
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=5000,stop_words='english')
vectors = cv.fit_transform(new_def['tags']).toarray()
vectors
vectors[0]
cv.get_feature_names_out()
from sklearn.metrics.pairwise import cosine_similarity
similarity = cosine_similarity(vectors)
similarity[0]
sorted(list(enumerate(similarity[0])),reverse='True',key=lambda x:x[1])[1:6]
def recommend(movie):
    movie_index = new_def[new_def['title'] == movie].index[0]
    distances = similarity[movie_index]
    movie_list = sorted(list(enumerate(distances)),reverse='True',key=lambda x:x[1])[1:6]
    for i in movie_list:
        print(new_def.iloc[i[0]].title)
recommend('Batman Begins')  
import pickle
pickle.dump(new_def,open('movies.pkl','wb'))  
new_def['title'].values   
pickle.dump(new_def.to_dict(),open('movies_dict.pkl','wb')) 
pickle.dump(similarity,open('similarity.pkl','wb'))
    