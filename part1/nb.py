from scipy.sparse import csr_matrix
from sklearn.naive_bayes import MultinomialNB


def fit(training_set, encoder, alpha=1.0):
    """
    Trains and returns a multinomial naive Bayes classifier.

    alpha: add-alpha smoothing (alpha=1 is Laplace smoothing)
    """
    X = csr_matrix([encoder(text) for text, label in training_set])
    y = [label for text, label in training_set]
    model = MultinomialNB(alpha=alpha)
    model.fit(X, y)
    return model


def predict(texts, model, encoder):
    X = csr_matrix([encoder(text) for text in texts])
    return model.predict(X)
