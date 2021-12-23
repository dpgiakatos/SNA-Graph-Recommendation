import pandas as pd
import numpy as np
from sklearn.preprocessing import Normalizer


class Dataset:
    """This class initialize our  dataset. Our dataset is MovieLens."""
    def __init__(self, directory):
        print('Loading local database...')
        self.links = pd.read_csv(directory + 'links.csv', sep=',', encoding='utf-8')
        self.movies = pd.read_csv(directory + 'movies.csv', sep=',', encoding='utf-8')
        self.ratings = pd.read_csv(directory + 'ratings.csv', sep=',', encoding='utf-8')
        self.tags = pd.read_csv(directory + 'tags.csv', sep=',', encoding='utf-8')
        print('Database loaded')

    def get_links(self):
        """Return a dataframe from the file links.csv."""
        return self.links

    def get_movies(self):
        """Return a dataframe from the file movies.csv"""
        return self.movies

    def get_ratings(self, normalize=False):
        """Return a dataframe from the file ratings.csv"""
        if normalize:
            norm = np.linalg.norm(self.ratings['rating'].to_numpy())
            self.ratings['rating'] = self.ratings['rating'].to_numpy()/norm
        return self.ratings

    def get_tags(self):
        """Return a dataframe from the file tags.csv"""
        return self.tags
