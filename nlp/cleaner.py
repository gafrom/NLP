import Stemmer
import re

class Cleaner(object):
  stemmer = Stemmer.Stemmer('russian')

  def __init__(self, stemming=True, locale='ru'):
    assert locale == 'ru'
    self._stemming = stemming

  def words(self, text):
    cleaned_words = self._clean(text).split(' ')
    if cleaned_words[0]  == '': cleaned_words = cleaned_words[1:]
    if len(cleaned_words) > 0 and cleaned_words[-1] == '': cleaned_words = cleaned_words[:-1]

    words = self._stem(cleaned_words) if self._stemming is True else cleaned_words

    return words

  def _clean(self, string):
    string = re.sub(r"[ёЁ]",               "е",     string)
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
