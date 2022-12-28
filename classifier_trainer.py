from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from nltk.stem.snowball import SnowballStemmer
import numpy as np
import sklearn.datasets
import glob
import pickle

training_data = []
tr_target = []

teesting_data = []
te_target = []

categories = ['awards', 'expansion', 'financing', 'production', 'others']

for category in categories:
    files = glob.glob('./dataset/%s/%s_*' % (category, category))
    fc = 0
    for file in files:
        print(f'Reading file : {file}')
        f = open(file, "r")
        if f:
            try:
                file_data = f.read()
                if fc < 30:
                    training_data.append(file_data)
                    tr_target.append(category)
                else:
                    teesting_data.append(file_data)
                    te_target.append(category)
                fc += 1
            except Exception:
                print(f'File : {file} Reading error')
                continue
            


train_dataset = sklearn.datasets.base.Bunch(data=training_data, target=tr_target)
test_dataset = sklearn.datasets.base.Bunch(data=teesting_data, target=te_target)


print('training data set %s : testing data set %s' % (len(train_dataset.data), len(test_dataset.data)))

stemmer = SnowballStemmer("english", ignore_stopwords=True)

class StemmedCountVectorizer(CountVectorizer):
    def build_analyzer(self):
        analyzer = super(StemmedCountVectorizer, self).build_analyzer()
        return lambda doc: ([stemmer.stem(w) for w in analyzer(doc)])

# Naive Bayes
text_clf_nb = Pipeline([
     ('vect', StemmedCountVectorizer(stop_words='english')),
     ('tfidf', TfidfTransformer()),
     ('clf', MultinomialNB(fit_prior=False))
 ])

text_clf_nb = text_clf_nb.fit(train_dataset.data, train_dataset.target)

predicted_nb = text_clf_nb.predict(test_dataset.data)
print(predicted_nb)
accuracy = np.mean(predicted_nb == test_dataset.target)

print('\nNaive Bayes Accuracy: %s' % accuracy)

# save trained model for later usage
pickle.dump(text_clf_nb, open("nb_txt_clf.sav", 'wb'))

print('\n\n')

# Support Vector Machine
text_clf_svm = Pipeline([
    ('vect', StemmedCountVectorizer(stop_words='english')),
    ('tfidf', TfidfTransformer()),
    ('clf-svm', SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, max_iter=5, random_state=42))]
)
_ = text_clf_svm.fit(train_dataset.data, train_dataset.target)

predicted_svm = text_clf_svm.predict(test_dataset.data)
print(predicted_svm)
accuracy = np.mean(predicted_svm == test_dataset.target)
print('\nSupport Vector Machine Accuracy : %s' % accuracy)


# save trained model for later usage
pickle.dump(text_clf_svm, open("svm_txt_clf.sav", 'wb'))
