import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.sparse import csr_matrix
import joblib
import os

class MovieRecommender:
    def __init__(self):
        self.movies = None
        self.ratings = None
        self.user_item_matrix = None
        self.svd = None
        self.model_file = 'movie_recommender.pkl'
        self.matrix_file = 'user_item_matrix.pkl'

    def load_data(self, movies_path='dataset/movies.csv', ratings_path='dataset/ratings.csv'):
        """Load and preprocess the data"""
        print("Loading data...")
        self.movies = pd.read_csv(movies_path)
        self.ratings = pd.read_csv(ratings_path)
        
        # Reduce dataset size for demo purposes
        self.ratings = self.ratings[self.ratings['userId'] <= 1000]
        movie_rating_counts = self.ratings['movieId'].value_counts()
        popular_movies = movie_rating_counts[movie_rating_counts >= 50].index
        self.ratings = self.ratings[self.ratings['movieId'].isin(popular_movies)]
        self.movies = self.movies[self.movies['movieId'].isin(self.ratings['movieId'])]

    def prepare_matrix(self):
        """Create user-item matrix"""
        print("Preparing user-item matrix...")
        movie_data = pd.merge(self.ratings, self.movies, on='movieId')
        self.user_item_matrix = movie_data.pivot_table(
            index='userId', 
            columns='title', 
            values='rating'
        ).fillna(0)
        
    def train_model(self, n_components=50):
        """Train SVD model"""
        print("Training model...")
        X = self.user_item_matrix.values
        X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)
        
        self.svd = TruncatedSVD(n_components=n_components, random_state=42)
        self.svd.fit(X_train)
        
        # Evaluate model
        X_pred = self.svd.inverse_transform(self.svd.transform(X_test))
        rmse = np.sqrt(mean_squared_error(X_test, X_pred))
        print(f"Model trained with RMSE: {rmse:.4f}")

    def get_recommendations(self, user_id, n_recommendations=5):
        """Generate movie recommendations for a user"""
        if user_id not in self.user_item_matrix.index:
            return self.get_popular_movies(n_recommendations)
            
        user_ratings = self.user_item_matrix.loc[user_id].values.reshape(1, -1)
        pred_ratings = self.svd.inverse_transform(self.svd.transform(user_ratings))
        pred_ratings = pd.Series(pred_ratings[0], index=self.user_item_matrix.columns)
        
        # Filter out already watched movies
        watched = self.user_item_matrix.loc[user_id][self.user_item_matrix.loc[user_id] > 0].index
        recommendations = pred_ratings.drop(watched).sort_values(ascending=False)
        
        return recommendations.head(n_recommendations)

    def get_popular_movies(self, n=5):
        """Get popular movies as fallback"""
        movie_stats = self.ratings.groupby('movieId').agg(
            avg_rating=('rating', 'mean'),
            num_ratings=('rating', 'count')
        ).reset_index()
        
        popular = movie_stats[movie_stats['num_ratings'] >= 100]
        popular_movies = popular.merge(self.movies, on='movieId')
        popular_movies = popular_movies.sort_values(['avg_rating', 'num_ratings'], ascending=False)
        
        return popular_movies[['title', 'avg_rating']].head(n)

    def save_model(self):
        """Save model and matrix for later use"""
        joblib.dump(self.svd, self.model_file)
        joblib.dump(self.user_item_matrix, self.matrix_file)
        print("Model saved successfully.")

    def load_saved_model(self):
        """Load previously saved model"""
        if os.path.exists(self.model_file) and os.path.exists(self.matrix_file):
            self.svd = joblib.load(self.model_file)
            self.user_item_matrix = joblib.load(self.matrix_file)
            print("Loaded saved model.")
            return True
        return False

    def visualize_top_movies(self, n=10):
        """Visualize top rated movies"""
        top_movies = self.ratings[self.ratings['rating'] == 5]['movieId'].value_counts().head(n)
        top_movies = top_movies.reset_index().merge(self.movies, left_on='index', right_on='movieId')
        
        plt.figure(figsize=(12, 8))
        sns.set_style("whitegrid")
        barplot = sns.barplot(x='count', y='title', data=top_movies, palette="viridis")
        
        plt.title(f'Top {n} Movies with Most 5-Star Ratings', fontsize=16)
        plt.xlabel('Number of 5-Star Ratings', fontsize=12)
        plt.ylabel('Movie Title', fontsize=12)
        
        for i, count in enumerate(top_movies['count']):
            barplot.text(count + 5, i, f'{count}', ha='left', va='center')
        
        plt.tight_layout()
        plt.show()

def main():
    recommender = MovieRecommender()
    
    if not recommender.load_saved_model():
        recommender.load_data()
        recommender.prepare_matrix()
        recommender.train_model()
        recommender.save_model()
    
    print("\nRecommendations for User 1:")
    print(recommender.get_recommendations(1))


if __name__ == "__main__":
    main()