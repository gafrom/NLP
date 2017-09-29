from nlp.preparer import Corpus

corpora_paths = ['./articles/good.articles', './articles/bad.articles'] 

for path in corpora_paths:
  corpus = Corpus(path)

  print(f"Corpus label: {corpus.label}")
  print(f"Corpus article: {corpus.raw[0]}")
  print(f"Corpus length: {len(corpus.articles)}")
  print(f"Corpus data: {corpus.articles[0]}")
  print(f"Corpus av length: {corpus.average_length()}")
