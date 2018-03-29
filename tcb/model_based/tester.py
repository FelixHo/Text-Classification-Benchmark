# encoding:utf-8
import os
import numpy as np
from beautifultable import BeautifulTable
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


def load_corpus(corpus_path):
    corpus = []
    labels = []  # 'Art', 'Enviornment', 'Space', 'Sports', 'Computer', 'Politics', 'Economy', 'Agriculture', 'History'

    with open(corpus_path, 'r') as f:
        for row in f:
            _content = row[row.index('@') + 1:]
            _label = row[:row.index('@')]
            corpus.append(_content)
            labels.append(_label)

    return corpus, labels


def feature_select(corpus, labels, k=1000):
    """
    select top k features through chi-square test
    """
    bin_cv = CountVectorizer(binary=True)
    le = LabelEncoder()
    X = bin_cv.fit_transform(corpus)
    y = le.fit_transform(labels).reshape(-1, 1)

    k = min(X.shape[1], k)
    skb = SelectKBest(chi2, k=k)
    skb.fit(X, y)

    feature_ids = skb.get_support(indices=True)
    feature_names = bin_cv.get_feature_names()
    vocab = {}

    for new_fid, old_fid in enumerate(feature_ids):
        feature_name = feature_names[old_fid]
        vocab[feature_name] = new_fid

    # we only care about the final extracted feature vocabulary
    return vocab


def feature_extract(corpus, labels, vocab):
    """
    feature extraction through TF-IDF
    """
    tfidf_vec = TfidfVectorizer(vocabulary=vocab)
    X = tfidf_vec.fit_transform(corpus)

    le = LabelEncoder()
    y = le.fit_transform(labels).reshape(-1)

    classes = le.classes_
    return X, y, classes


def init_estimators():
    return [
        {'classifier': 'NB', 'model': MultinomialNB()},
        {'classifier': 'LR', 'model': LogisticRegression(random_state=42)},
        {'classifier': 'L-SVM', 'model': LinearSVC(max_iter=1000, random_state=42)},
        {'classifier': 'RBF-SVM', 'model': SVC(max_iter=1000, random_state=42)},
        {'classifier': 'RF', 'model': RandomForestClassifier(n_estimators=100, random_state=42)},
        {'classifier': 'XGB', 'model': XGBClassifier(n_estimators=100, random_state=42)},
        {'classifier': 'LGBM', 'model': LGBMClassifier(n_estimators=100, random_state=42)},
    ]


def get_cv_scores(estimator, X, y, scoring, cv=5):
    return cross_validate(estimator, X, y, cv=cv, n_jobs=1,
                          scoring=scoring,
                          return_train_score=False)


if __name__ == '__main__':
    corpus_path = os.path.join(os.path.dirname(__file__), '../../data/corpus/FDU_NLP_corpus_seg_balanced.txt')
    corpus, label = load_corpus(corpus_path)
    vocab = feature_select(corpus, label, k=1000)
    X, y, cls = feature_extract(corpus, label, vocab)

    # from sklearn.preprocessing import StandardScaler
    # X = StandardScaler().fit_transform(X.toarray())

    ## prevent lose their sparsity
    # X = StandardScaler(with_mean=False).fit_transform(X.toarray())

    benchmark = BeautifulTable(max_width=100)
    benchmark.column_headers = [
        'classifier',
        'fit_time',
        'score_time',
        'test_precision_micro',
        'test_recall_micro',
        'test_f1_micro'
    ]

    estimators = init_estimators()

    for estimator in estimators:
        print '### %s ###' % estimator['classifier']
        row = [estimator['classifier'], ]
        scores = get_cv_scores(estimator['model'], X, y, scoring=['precision_micro', 'recall_micro', 'f1_micro'])

        for k in benchmark.column_headers[1:]:
            row.append(np.mean(scores[k]))

        benchmark.append_row(row)

    print benchmark
