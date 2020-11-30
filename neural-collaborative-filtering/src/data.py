import torch
import random
import pandas as pd
from copy import deepcopy
from torch.utils.data import DataLoader, Dataset

random.seed(0)


class UserItemRatingDataset(Dataset):
    """Wrapper, convert <user, item, rating> Tensor into Pytorch Dataset"""
    def __init__(self, user_tensor, item_tensor, target_tensor):
        """
        args:

            target_tensor: torch.Tensor, the corresponding rating for <user, item> pair
        """
        self.user_tensor = user_tensor
        self.item_tensor = item_tensor
        self.target_tensor = target_tensor

    def __getitem__(self, index):
        return self.user_tensor[index], self.item_tensor[index], self.target_tensor[index]

    def __len__(self):
        return self.user_tensor.size(0)


class SampleGenerator(object):
    """Construct dataset for NCF"""

    def __init__(self, ratings, jokes):
        """
        args:
            ratings: pd.DataFrame, which contains 4 columns = ['userId', 'itemId', 'rating', 'timestamp']
        """
        assert 'userId' in ratings.columns
        assert 'itemId' in ratings.columns
        assert 'itemEmbedding' in ratings.columns
        assert 'rating' in ratings.columns

        self.ratings = ratings
        self.jokes = jokes
        # explicit feedback using _normalize and implicit using _binarize
        # self.preprocess_ratings = self._normalize(ratings)
        self.preprocess_ratings = self._binarize(ratings)
        self.user_pool = set(self.ratings['userId'].unique())
        self.item_pool = set(self.ratings['itemId'].unique())
        # create negative item samples for NCF learning
        self.negatives = self._sample_negative(ratings, jokes)
        self.train_ratings, self.test_ratings = self._split_loo(self.preprocess_ratings, jokes)

    def _normalize(self, ratings):
        """normalize into [0, 1] from [0, max_rating], explicit feedback"""
        ratings = deepcopy(ratings)
        max_rating = ratings.rating.max()
        ratings['rating'] = ratings.rating * 1.0 / max_rating
        return ratings
    
    def _binarize(self, ratings):
        """binarize into 0 or 1, imlicit feedback"""
        ratings = deepcopy(ratings)
        ratings['rating'][ratings['rating'] > 0] = 1.0 # positive rating
        ratings['rating'][ratings['rating'] < 0] = 0.0 # negative rating
        return ratings

    def _split_loo(self, ratings, jokes):
        """leave one out train/test split """
        # ratings['rank_latest'] = ratings.groupby(['userId'])['timestamp'].rank(method='first', ascending=False)
        ratings['rank_latest'] = ratings.groupby(['userId'])['itemId'].rank(method='first', ascending=False)
        test = ratings[ratings['rank_latest'] == 1]
        train = ratings[ratings['rank_latest'] > 1]
        assert train['userId'].nunique() == test['userId'].nunique()
        return train[['userId', 'itemId', 'itemEmbedding', 'rating']], test[['userId', 'itemId', 'itemEmbedding', 'rating']]

    def _sample_negative(self, ratings, jokes):
        """return all negative items & 100 sampled negative items"""
        interact_status = ratings.groupby('userId')['itemId'].apply(set).reset_index().rename(
            columns={'itemId': 'interacted_items'})
        interact_status['negative_items'] = interact_status['interacted_items'].apply(lambda x: self.item_pool - x) # negative items are the items without ratings by given user
        
        n_nagatives_items = interact_status['negative_items'].apply(lambda x: len(x))
        interact_status = interact_status[n_nagatives_items != 0] # drop all users with 0 negative items
        min_n_negatives = interact_status['negative_items'].apply(lambda x: len(x)).min()
        min_n_negatives = 50 if min_n_negatives>50 else min_n_negatives
        interact_status['negative_samples'] = interact_status['negative_items'].apply(lambda x: random.sample(x, min_n_negatives)) # negative samples are selected randomly from negative items
        
        interact_status['negative_items_embeddings'] = interact_status['negative_items'].apply(lambda x: list(jokes[jokes["joke id"].isin(list(x))]['embedding']))
        interact_status['negative_samples_embeddings'] = interact_status['negative_samples'].apply(lambda x: list(jokes[jokes["joke id"].isin(list(x))]['embedding']))

        return interact_status[['userId', 'negative_items', 'negative_items_embeddings', 'negative_samples', 'negative_samples_embeddings']]

    def instance_a_train_loader(self, num_negatives, batch_size):
        """instance train loader for one training epoch"""
        users, items, ratings = [], [], []
        train_ratings = pd.merge(self.train_ratings, self.negatives[['userId', 'negative_items_embeddings']], on='userId')
        train_ratings['negatives'] = train_ratings['negative_items_embeddings'].apply(lambda x: random.sample(x, num_negatives))
        for row in train_ratings.itertuples():
            users.append(int(row.userId))
            items.append(row.itemEmbedding)
            ratings.append(float(row.rating))
            for i in range(num_negatives):
                users.append(int(row.userId))
                items.append(row.negatives[i])
                ratings.append(float(0))  # negative samples get 0 rating
        dataset = UserItemRatingDataset(user_tensor=torch.LongTensor(users),
                                        item_tensor=torch.FloatTensor(items),
                                        target_tensor=torch.FloatTensor(ratings))
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)

    @property
    def evaluate_data(self):
        """create evaluate data"""
        test_ratings = pd.merge(self.test_ratings, self.negatives[['userId', 'negative_samples', 'negative_samples_embeddings']], on='userId')
        test_users, test_items, negative_users, negative_items = [], [], [], []
        for row in test_ratings.itertuples():
            test_users.append(int(row.userId))
            test_items.append(row.itemEmbedding)
            for i in range(len(row.negative_samples)):
                negative_users.append(int(row.userId))
                negative_items.append(row.negative_samples_embeddings[i])

        return [torch.LongTensor(test_users), torch.FloatTensor(test_items), torch.LongTensor(negative_users),
                torch.FloatTensor(negative_items)]
