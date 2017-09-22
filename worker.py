from nlp.preparer import Corpus

corpus = Corpus('./articles/good.articles')

print(f"Corpus label: {corpus.label}")
print(f"Corpus article: {corpus.raw[0]}")
print(f"Corpus data: {corpus.articles[0]}")
print(f"Corpus av length: {corpus.average_length()}")
