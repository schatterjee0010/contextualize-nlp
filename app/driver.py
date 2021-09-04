import os

from ml_models import gaussian_nb_model, multinomial_nb_model, decision_tree_classifier_model
from ml_models.preprocessor import nlp_preprocessor


class Driver:
    def __init__(self, fl_loc, fl_type, fl_delim, model_type):
        self.fl_loc = fl_loc
        self.fl_type = fl_type
        self.fl_delim = fl_delim
        self.model_type = model_type
        self.nlp_preprocessor = nlp_preprocessor.NlpPreprocessor(max_features=12000,
                                                                 num_of_record=10000)
        self.model = None
        if model_type == 'GAUSSIAN':
            self.model = gaussian_nb_model.GaussianNbModel()
        elif model_type == 'MULTINOMIAL':
            self.model = multinomial_nb_model.MultinomialNbModel()
        elif model_type == 'DTREECLSF':
            self.model = decision_tree_classifier_model.DecisionTreeClassifierModel()

    def handler(self):
        self.nlp_preprocessor.load_file(self.fl_loc, self.fl_type, self.fl_delim)
        self.nlp_preprocessor.extract_features(cols=['Review Text'])
        self.nlp_preprocessor.extract_target(col='Rating', custom_flg=True)
        self.nlp_preprocessor.lower_clean(col='Review Text')
        X_train, X_test, y_train, y_test = self.nlp_preprocessor.gen_train_test()
        self.model.train_n_predict(X_train, X_test, y_train)
        self.model.predict_accuracy(y_test)


if __name__ == '__main__':
    fl_loc = os.environ['FL_LOC']
    fl_type = "CSV"
    fl_delim = ","
    model_type = os.environ['MODEL_TYPE']

    Driver(fl_loc, fl_type, fl_delim, model_type).handler()
