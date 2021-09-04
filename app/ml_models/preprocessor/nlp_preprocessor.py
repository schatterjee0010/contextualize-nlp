import pandas as pd
import numpy as np
from nltk.stem.porter import PorterStemmer
from copy import deepcopy
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split


class NlpPreprocessor:
    def __init__(self, max_features: int, num_of_record: int):
        self.dataset = None
        self.features = None
        self.target = None
        self.corpus = []
        self.num_of_record = num_of_record
        self.porter_stemmer = PorterStemmer()
        self.count_vector = CountVectorizer(max_features=max_features)

    def load_file(self, fl_loc: str, fl_type: str, fl_delim: str):

        if fl_type == 'CSV' and fl_delim == ',':
            self.dataset = pd.read_csv(fl_loc)
        print("File loaded")

    def extract_features(self, cols: list):
        self.features = self.dataset[cols]
        print("Features extracted")

    def extract_target(self, col: str, custom_flg: bool):

        if custom_flg:
            self.target = np.where(self.dataset[col] > 3, 1, 0)
        else:
            self.target = self.dataset[col]
        print("Target extracted")

    def lower_clean(self, col: str):
        self.corpus = self.features[col].str.lower().replace(
            to_replace="[^a-zA-Z0-9 \n\.]",
            value=" ",
            regex=True
        ).values.astype('U')
        print("Cleaned up")

    def apply_stemming(self):
        corpus_interim = []
        for sentence in self.corpus[:self.num_of_record]:
            words = sentence.split()
            line = [self.porter_stemmer.stem(word) for word in words]
            line = ' '.join(line)
            corpus_interim.append(line)
        self.corpus = deepcopy(corpus_interim)
        print("Applied Stemming")

    def gen_train_test(self):
        X = self.count_vector.fit_transform(self.corpus[:self.num_of_record]).toarray()
        y = self.target[:self.num_of_record]
        print(X)
        print(y)
        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                            test_size=0.2,
                                                            random_state=42)
        return X_train, X_test, y_train, y_test
