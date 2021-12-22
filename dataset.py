import pandas as pd


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

    def get_ratings(self):
        """Return a dataframe from the file ratings.csv"""
        return self.ratings

    def get_tags(self):
        """Return a dataframe from the file tags.csv"""
        return self.tags
