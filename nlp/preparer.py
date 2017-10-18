import re

class Corpus(object):
  def __init__(self, filename):
    self._filename = filename
    self.length = None
    self.clener = Cleaner()

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
    # set label `good` based on `good.articles` filename pattern
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
      data.append(self.cleaner.words(article))

    return data
