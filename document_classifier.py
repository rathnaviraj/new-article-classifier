from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem.snowball import SnowballStemmer
import pickle
import sys


stemmer = SnowballStemmer("english", ignore_stopwords=True)

class StemmedCountVectorizer(CountVectorizer):
    def build_analyzer(self):
        analyzer = super(StemmedCountVectorizer, self).build_analyzer()
        return lambda doc: ([stemmer.stem(w) for w in analyzer(doc)])

stemmed_count_vect = StemmedCountVectorizer(stop_words='english')

print ('\n==== Select Classification Model ====\n\n')

print ('Naive Bayes - nb')
print ('Support Vector Machine - svm')

code = input("\nEnter Model Code (nb/svm) : ")

try:
    loaded_model = pickle.load(open(f'{code}_txt_clf.sav', 'rb'))
except Exception:
    sys.exit('Error: specified model not found')

document = None

while True:
    file_path = input("\nDocument Path : ")
    f = open(file_path, "r")
    if f:
        document = f.read()
        print(document.lower())
        predicted = loaded_model.predict([document.lower()])
        print(predicted[0])
