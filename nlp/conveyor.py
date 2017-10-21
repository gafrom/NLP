import random
import collections

# Here by `pool` we mean smth like that:
# pool = [(words, label), (words, label), (words, label), ...]
class Conveyor(object):
  def __init__(self, pool, size):
    self.size = size
    self.pool = pool
    self.belt = collections.deque()

  def next(self):
    self._load_more_if_needed()

    feed = ([], [], [])
    for _ in range(self.size):
      bundle = self.belt.popleft()
      feed[0].append(bundle[0]) # data
      feed[1].append(bundle[1]) # label
      feed[2].append(bundle[2]) # state clearer

    return feed

  def rewind(self):
    self.belt = collections.deque()

  def _load_more_if_needed(self):
    while len(self.belt) < self.size:
      data, label = random.choice(self.pool)
      bundle = (data[0], label, 0.)
      self.belt.append(bundle)
      for item in data[1:]:
        bundle = (item, label, 1.)
        self.belt.append(bundle)

