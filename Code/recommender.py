import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

class HybridRecommender:
    def __init__(self):
        self.user_item_matrix = None
        self.item_features = None
        self.item_similarity = None

    def prepare_data(self, interactions_df, item_features_df):
        self.user_item_matrix = interactions_df.pivot(
            index='user_id', columns='item_id', values='interaction_strength'
        ).fillna(0)
        self.item_features = item_features_df

    def compute_item_similarity(self):
        tfidf = TfidfVectorizer()
        item_features_tfidf = tfidf.fit_transform(self.item_features['description'])
        self.item_similarity = cosine_similarity(item_features_tfidf)

    def collaborative_filtering(self, user_id, n_recommendations=5):
        user_vector = self.user_item_matrix.loc[user_id].values.reshape(1, -1)
        similarities = cosine_similarity(user_vector, self.user_item_matrix.values)
        similar_users = similarities.argsort()[0][::-1][1:6]  # Top 5 similar users
        
        recommendations = self.user_item_matrix.iloc[similar_users].mean().sort_values(ascending=False)
        return recommendations.head(n_recommendations)

    def content_based_filtering(self, item_id, n_recommendations=5):
        item_index = self.item_features.index.get_loc(item_id)
        similar_items = self.item_similarity[item_index].argsort()[::-1][1:n_recommendations+1]
        return self.item_features.iloc[similar_items]

    def hybrid_recommendations(self, user_id, n_recommendations=5):
        cf_recs = self.collaborative_filtering(user_id, n_recommendations)
        hybrid_recs = []
        
        for item_id in cf_recs.index:
            cb_recs = self.content_based_filtering(item_id, 2)
            hybrid_recs.extend([item_id] + cb_recs.index.tolist())
        
        return pd.Series(hybrid_recs).drop_duplicates().head(n_recommendations)

if __name__ == "__main__":
    # Example usage
    interactions_df = pd.read_csv("data/processed/user_item_interactions.csv")
    item_features_df = pd.read_csv("data/processed/item_features.csv")
    
    recommender = HybridRecommender()
    recommender.prepare_data(interactions_df, item_features_df)
    recommender.compute_item_similarity()
    
    user_id = 12345  # Example user ID
    recommendations = recommender.hybrid_recommendations(user_id)
    print(f"Top 5 recommendations for user {user_id}:")
    print(recommendations)
