import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score,\
    precision_recall_fscore_support, classification_report
from sklearn.naive_bayes import MultinomialNB


class MultinomialNbModel:

    def __init__(self):
        self.classifier = MultinomialNB()
        self.y_pred = None

    def _fit(self, X_train, y_train):
        print("Training Multinomial Model")
        self.classifier.fit(X_train, y_train)

    def train_n_predict(self, X_train, X_test, y_train):
        self._fit(X_train, y_train)
        self.y_pred = self.classifier.predict(X_test)
        return self.y_pred

    def predict_freq(self):
        unique, count = np.unique(self.y_pred, return_counts=True)
        freq = np.asarray((unique, count)).T
        print(f"Frequencies: {freq}")

    def predict_accuracy(self, y_actual):
        cm = confusion_matrix(y_actual, self.y_pred)
        print(f"Confusion Matrix:\n{cm}")
        score = accuracy_score(y_actual, self.y_pred)
        print(f"Accuracy Score: {score}")
        print(f"Classification Report:\n{classification_report(y_actual, self.y_pred)}")
        return score

    def perf_metrics(self, y_actual):
        precision, recall, fscore, _ = precision_recall_fscore_support(
            y_actual,
            self.y_pred, average='weighted')
        print(f" Precision: {precision},\n Recall: {recall},\n FScore: {fscore}")
