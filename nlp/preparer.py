from nlp.cleaner import Cleaner
import re

class Corpus(object):
  def __init__(self, filename, max_words_per_article=None, min_words_per_article=4,
                     truncating=True, stemming=True):
    self._filename   = filename
    self._max_length = max_words_per_article
    self._min_length = min_words_per_article
    self._truncating = truncating
    self.length      = None
    self.cleaner     = Cleaner(stemming)

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
    fit_by_limit = self._truncate_by_limit if self._truncating is True else self._split_by_limit

    data = []
    for article in self.raw:
      words = self.cleaner.words(article)
      data += fit_by_limit(words)

    return data

  def _truncate_by_limit(self, collection):
    if len(collection) < self._min_length: return []
    if self._max_length is None: return [collection]

    temp   = []
    for i, item in enumerate(collection):
      temp.append(item)
      if item == '.' and i > self._max_length: break

    return [temp]

  def _split_by_limit(self, collection):
    if len(collection) < self._min_length: return []
    if self._max_length is None: return [collection]

    bottom = self._max_length / 4

    result = []
    temp   = []
    for i, item in enumerate(collection):
      temp.append(item)
      if item == '.' and i > (len(result) + 1) * self._max_length:
        if len(collection) - i < bottom:
          temp += collection[i + 1:]
          break
        result.append(temp)
        temp = []
    result.append(temp)

    return result
