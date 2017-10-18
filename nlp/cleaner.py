import Stemmer
import re

class Cleaner(object):
  stemmer = Stemmer.Stemmer('russian')

  def __init__(self, locale='ru'):
    assert locale == 'ru'

  def words(self, text):
    cleaned_words = self._clean(text).split(' ')
    stemmed_words = self._stem(cleaned_words)

    return stemmed_words

  def _clean(self, string):
    string = re.sub(r"[^А-Яа-я0-9(),!?.]", " ",     string)
    string = re.sub(r"\d+(\.|,)?\d*",      " num ", string)
    string = re.sub(r"\.",                 " . ",   string)
    string = re.sub(r",",                  " , ",   string)
    string = re.sub(r"!",                  " ! ",   string)
    string = re.sub(r"\(",                 " ( ",   string)
    string = re.sub(r"\)",                 " ) ",   string)
    string = re.sub(r"\?",                 " ? ",   string)
    string = re.sub(r"\s{2,}",             " ",     string)
    return string.strip().lower()

  def _stem(self, words):
    return self.stemmer.stemWords(words)
