import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

# Load and preprocess data
@st.cache_data
def load_data():
    data = pd.read_csv("C:/Degree Year 2 Semester 3/AI/Music dataset/music_sentiment_dataset_with_ratings.csv")
    
    # Column normalization
    column_mapping = {
        'tempo': 'Tempo',
        'energy': 'Energy',
        'user_id': 'User_ID',
        'song_name': 'Song_Name',
        'sentiment': 'Sentiment_Label'
    }
    data = data.rename(columns={k: v for k, v in column_mapping.items() if k in data.columns})
    
    # Convert categorical features to numerical
    categorical_mapping = {
        'Tempo': {'Low': 1, 'Medium': 2, 'High': 3, 'low': 1, 'medium': 2, 'high': 3},
        'Energy': {'Low': 1, 'Medium': 2, 'High': 3, 'low': 1, 'medium': 2, 'high': 3}
    }
    
    for col in ['Tempo', 'Energy']:
        if col in data.columns:
            if data[col].dtype == object:
                data[col] = data[col].map(categorical_mapping.get(col, {})).fillna(0)
            data[col] = pd.to_numeric(data[col], errors='coerce').fillna(0)
    
    return data

music_data = load_data()

# Initialize session state
if 'recommended_songs' not in st.session_state:
    st.session_state.recommended_songs = set()

# Feature preprocessing
scaler = MinMaxScaler()
tfidf = TfidfVectorizer(stop_words='english')

def preprocess_features(data):
    available_features = []
    
    # Text features
    text_columns = []
    if 'Genre' in data.columns:
        text_columns.append('Genre')
    if 'Artist' in data.columns:
        text_columns.append('Artist')
    
    if text_columns:
        text_data = data[text_columns].fillna('').apply(lambda x: ' '.join(x), axis=1)
        text_matrix = tfidf.fit_transform(text_data)
        available_features.append(text_matrix)
    
    # Numerical features
    num_columns = []
    if 'rating' in data.columns:
        num_columns.append('rating')
    for col in ['Tempo', 'Energy']:
        if col in data.columns:
            num_columns.append(col)
    
    if num_columns:
        num_data = data[num_columns].fillna(0)
        num_matrix = scaler.fit_transform(num_data)
        available_features.append(num_matrix)
    
    return hstack(available_features) if len(available_features) > 1 else available_features[0]

# Recommendation functions
def content_based_recommendations(mood, user_id, num_recs):
    try:
        mood_data = music_data[music_data['Sentiment_Label'] == mood]
        if mood_data.empty:
            return pd.DataFrame()
        
        features = preprocess_features(mood_data)
        similarities = cosine_similarity(features)
        
        user_rated = music_data[music_data['User_ID'] == user_id]['Song_Name'].tolist()
        user_ratings = music_data[(music_data['User_ID'] == user_id) & (music_data['Sentiment_Label'] == mood)]
        
        if not user_ratings.empty:
            indices = user_ratings.index.tolist()
            sim_scores = np.mean(similarities[indices], axis=0)
        else:
            sim_scores = np.mean(similarities, axis=0)
        
        top_indices = np.argsort(sim_scores)[::-1]
        recommendations = mood_data.iloc[top_indices]
        recommendations = recommendations[~recommendations['Song_Name'].isin(user_rated)]
        return recommendations.head(num_recs).drop_duplicates(subset=['Song_Name'])
    
    except Exception as e:
        st.error(f"Content-based recommendation error: {str(e)}")
        return pd.DataFrame()

def collaborative_filtering_recommendations(user_id, num_recs):
    try:
        if 'User_ID' not in music_data.columns or 'rating' not in music_data.columns:
            return pd.DataFrame()
        
        user_item_matrix = music_data.pivot_table(index='User_ID', columns='Song_Name', values='rating', fill_value=0)
        similarities = cosine_similarity(user_item_matrix)
        
        similar_users = np.argsort(similarities[user_id-1])[::-1][1:4]
        recommendations = pd.DataFrame()
        
        for sim_user in similar_users:
            sim_user_ratings = user_item_matrix.iloc[sim_user]
            top_songs = sim_user_ratings[sim_user_ratings > 3].index.tolist()
            new_recs = music_data[music_data['Song_Name'].isin(top_songs)]
            recommendations = pd.concat([recommendations, new_recs])
        
        user_rated = music_data[music_data['User_ID'] == user_id]['Song_Name'].tolist()
        recommendations = recommendations[~recommendations['Song_Name'].isin(user_rated)]
        return recommendations.drop_duplicates(subset=['Song_Name']).head(num_recs)
    
    except Exception as e:
        st.error(f"Collaborative filtering error: {str(e)}")
        return pd.DataFrame()

def hybrid_recommendations(user_id, mood, num_recs):
    content_recs = content_based_recommendations(mood, user_id, num_recs)
    collab_recs = collaborative_filtering_recommendations(user_id, num_recs)
    hybrid = pd.concat([content_recs, collab_recs]).drop_duplicates(subset=['Song_Name'])
    return hybrid.head(num_recs)

# Streamlit UI
st.title("🎵Music Recommendation System")

# User inputs
user_id = st.number_input("User ID", min_value=1, max_value=100, value=1)
mood_options = music_data['Sentiment_Label'].unique().tolist() if 'Sentiment_Label' in music_data.columns else ["Happy", "Sad"]
mood = st.selectbox("Current Mood", mood_options)
rec_type = st.selectbox("Recommendation Type", ["Content-Based", "Collaborative", "Hybrid"])
num_recs = st.slider("Number of Recommendations", 1, 10, 5)

# Recommendation display
if st.button("Get Recommendations"):
    with st.spinner("Analyzing your music preferences..."):
        try:
            if rec_type == "Content-Based":
                recs = content_based_recommendations(mood, user_id, num_recs)
            elif rec_type == "Collaborative":
                recs = collaborative_filtering_recommendations(user_id, num_recs)
            else:
                recs = hybrid_recommendations(user_id, mood, num_recs)
            
            st.session_state.recommended_songs = set(recs['Song_Name'].tolist()) if not recs.empty else set()
            
            if not recs.empty:
                st.subheader(f"Recommended Songs for {mood} Mood")
                for idx, row in recs.iterrows():
                    with st.container():
                        st.markdown(f"**{row['Song_Name']}**  \n*by {row.get('Artist', 'Unknown Artist')}*")
                        details = []
                        if 'Genre' in row: details.append(f"**Genre:** {row['Genre']}")
                        if 'rating' in row: details.append(f"**Avg Rating:** {row['rating']:.1f}/5")
                        if 'Energy' in row: details.append(f"**Energy:** {row['Energy']}")
                        st.markdown("  \n".join(details))
                        st.markdown("---")
            else:
                st.warning("No recommendations found. Please try different settings.")
        
        except Exception as e:
            st.error(f"Recommendation system error: {str(e)}")

# Sidebar
st.sidebar.header("User Dashboard")
if st.sidebar.checkbox("Show Dataset Summary"):
    st.sidebar.write("Total Songs:", len(music_data))
    st.sidebar.write("Unique Artists:", music_data['Artist'].nunique() if 'Artist' in music_data.columns else "N/A")
    st.sidebar.write("Average Rating:", f"{music_data['rating'].mean():.1f}" if 'rating' in music_data.columns else "N/A")

# Dataset info
st.sidebar.markdown("---")
st.sidebar.subheader("Dataset Information")
if st.sidebar.checkbox("Show Column Names"):
    st.sidebar.write(music_data.columns.tolist())
if st.sidebar.checkbox("Show Sample Data"):
    st.sidebar.dataframe(music_data.head(3))

# Additional Suggestions: Spotify Integration for Song Previews
def get_spotify_preview(song_name, artist_name):
    sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id='your-client-id', client_secret='your-client-secret'))
    result = sp.search(q=f"track:{song_name} artist:{artist_name}", type='track', limit=1)
    if result['tracks']['items']:
        preview_url = result['tracks']['items'][0]['preview_url']
        return preview_url
    return None
