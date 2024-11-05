import numpy as np
from collections import defaultdict

class NewsValueMaximizer:
    def __init__(self, articles, aligned_articles):
        self.articles = articles
        self.aligned_articles = aligned_articles
        self.k = len(articles)
        self.rewards = np.zeros(self.k)
        self.update_rewards()
        self.exploration_rate = 0.2
        self.action_counts = np.zeros(self.k)
        
    def update_rewards(self):
        for i, article in enumerate(self.articles):
            if article in self.aligned_articles:
                self.rewards[i] = 1.1
            else:
                self.rewards[i] = 1.0
    
    def recommend_article(self):
        if np.random.rand() < self.exploration_rate:
            return np.random.randint(self.k)
        else:
            return np.argmax(self.rewards)
    
    def update_article_reward(self, article_idx, interaction_type):
        self.action_counts[article_idx] += 1
        if interaction_type == "positive":
            self.rewards[article_idx] += 1.0 / self.action_counts[article_idx]
        else:
            self.rewards[article_idx] -= 1.0 / self.action_counts[article_idx]

# Example usage
articles = ["article_1", "article_2", "article_3", "article_4", "article_5"]
aligned_articles = ["article_1", "article_3", "article_5"]

maximizer = NewsValueMaximizer(articles, aligned_articles)

# Simulate user interactions
for _ in range(100):
    article_idx = maximizer.recommend_article()
    article = maximizer.articles[article_idx]
    interaction_type = "positive" if np.random.rand() < 0.7 else "negative"
    maximizer.update_article_reward(article_idx, interaction_type)
    print(f"Recommended article: {article}, Interaction type: {interaction_type}")
