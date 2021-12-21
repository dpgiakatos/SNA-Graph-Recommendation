from sklearn.feature_extraction.text import TfidfVectorizer


class Embedding:
    def __init__(self, method='tf'):
        self.corpus = ['Action', 'Adventure', 'Animation', 'Children', 'Comedy', 'Crime', 'Documentary', 'Drama',
                       'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller',
                       'War', 'Western', '(no genres listed)']
        self.method = method
        if method == 'tf-idf':
            self.tfidf = TfidfVectorizer()
            self.tfidf.fit(self.corpus)

    def transform(self, document):
        if self.method == 'tf':
            return [1 if value in document else 0 for value in self.corpus]
        elif self.method == 'tf-idf':
            return self.tfidf.transform([document])
