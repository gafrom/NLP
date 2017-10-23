from nlp.cleaner import Cleaner
import re

class Corpus(object):
  def __init__(self, filename, max_words_per_article=None, min_words_per_article=4):
    self._filename   = filename
    self._max_length = max_words_per_article
    self._min_length = min_words_per_article
    self._bottom     = self._max_length / 4
    self.length      = None
    self.cleaner     = Cleaner()

    self.label    = self._set_label()
    self.raw      = self._load()
    self.articles = self._articles()

  def average_length(self):
    if self.length is not None: return self.length

    lenghts = []
    for article in self.articles:
      lenghts.append(len(article))

    self.length = sum(lenghts)/len(lenghts)
    return self.length

  def _set_label(self):
    # extract label `my_label` from `my_label.articles` filename pattern
    less_extension = re.split('[\.]', self._filename)[-2]
    label = re.split('[\/]', less_extension)[-1]

    return label

  def _load(self):
    articles = []
    with open(self._filename) as f:
      for line in f: articles.append(line)

    return articles

  def _articles(self):
    data = []
    for article in self.raw:
      words = self.cleaner.words(article)
      data += self._split_by_limit(words)

    return data

  def _split_by_limit(self, collection):
    if len(collection) < self._min_length: return []
    if self._max_length is None: return [collection]

    result = []
    temp   = []
    for i, item in enumerate(collection):
      temp.append(item)
      if item == '.' and i > (len(result) + 1) * self._max_length:
        if len(collection) - i < self._bottom:
          temp += collection[i + 1:]
          break
        result.append(temp)
        temp = []
    result.append(temp)

    return result
