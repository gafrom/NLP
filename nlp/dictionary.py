from pathlib import Path
import pickle

class Dictionary(object):
  def __init__(self, path):
    self.path = path

  def load(self):
    dictionary_file_name = f"{self.path}.dictionary"
    if Path(dictionary_file_name).is_file():
      with open(dictionary_file_name, 'rb') as fp:
        return pickle.load(fp)
    else:
      msg = f"Cannot find stored dictionary file: {dictionary_file_name}"
      raise FileNotFoundError(msg)
