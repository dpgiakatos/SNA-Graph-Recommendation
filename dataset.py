import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class Dataset:
    """This class initialize our  dataset. Our dataset is MovieLens."""
    def __init__(self, directory):
        print('Loading local database...')
        self.links = pd.read_csv(directory + 'links.csv', sep=',', encoding='utf-8')
        self.movies = pd.read_csv(directory + 'movies.csv', sep=',', encoding='utf-8')
        self.ratings = pd.read_csv(directory + 'ratings.csv', sep=',', encoding='utf-8')
        self.tags = pd.read_csv(directory + 'tags.csv', sep=',', encoding='utf-8')
        self.connection_between_users()
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

    @staticmethod
    def bin_array(num, m):
        """Convert a positive integer num into an m-bit bit vector"""
        return np.array(list(np.binary_repr(num).zfill(m))).astype(np.int8)

    def connection_between_users(self):

        result = []
        for value in self.ratings['movieId']:
            r = Dataset.bin_array(value, 6)
            num = int(''.join(map(str, r)))
            result.append(num)

        self.ratings['newmovieId'] = result

        df2 = self.ratings[self.ratings.duplicated('userId', keep=False)].groupby('userId')['newmovieId'].apply(list).reset_index()

        listusers = []
        for i in range(len(df2)):
            j = i + 1
            for j in range(len(df2) - i):
                if len(df2['newmovieId'][i]) != len(df2['newmovieId'][j]):
                    if len(df2['newmovieId'][i]) < len(df2['newmovieId'][j]):
                        l = len(df2['newmovieId'][j]) - len(df2['newmovieId'][i])
                        df2['newmovieId'][i].extend([0] * (l))
                    else:
                        l = len(df2['newmovieId'][i]) - len(df2['newmovieId'][j])
                        df2['newmovieId'][j].extend([0] * (l))
                c = cosine_similarity([df2['newmovieId'][i]], [df2['newmovieId'][j]])
                listusers.append([df2['userId'][i], df2['userId'][j], round(c.flat[0], 3)])
        df = pd.DataFrame(listusers, columns=['User1', 'User2', 'Similarity'])
        df = df[df['User1'] != df['User2']]
        df2 = df[df['Similarity'] > 0.7]
        self.similarityusers = df2

    def get_similarity_users(self):
        return self.similarityusers

