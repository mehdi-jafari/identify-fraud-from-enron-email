# import nltk
# nltk.download()

from nltk.corpus import stopwords
sw = stopwords.words("english")
print len(sw)

from nltk.stem.snowball import SnowballStemmer
stemmer  = SnowballStemmer("english")
print stemmer.stem("responsiveness")
