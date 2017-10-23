# Usage example
# =============
#
# batch_size = 3
# a = [['A', 'B', 'C', 'D'], ['E', 'F', 'G', 'H'], ['I'], ['J', 'K', 'L']]
# b = [['a', 'b', 'c'], ['d', 'e'], ['f', 'g', 'h']]

# print(f'Good articles({len(a)}): {a}')
# print(f'Bad  articles({len(b)}): {b}')

# batches = BatchIterator(batch_size, a, b, 0, 1)

# print(f'\nGenerated batches of length {batch_size}:')
# for batch in batches:
#   print(batch)
#
#
# Output
# ======
#
# Good articles(4): [['A', 'B', 'C', 'D'], ['E', 'F', 'G', 'H'], ['I'], ['J', 'K', 'L']]
# Bad  articles(3): [['a', 'b', 'c'], ['d', 'e'], ['f', 'g', 'h']]

# Generated batches of length 3:
# [['A', 'B', 'C'], [0, 0, 0], [1.0, 1.0, 1.0]]
# [['D', 'a', 'b'], [0, 1, 1], [1.0, 0.0, 1.0]]
# [['c', 'E', 'F'], [1, 0, 0], [1.0, 0.0, 1.0]]
# [['G', 'H', 'd'], [0, 0, 1], [1.0, 1.0, 0.0]]
# [['e', 'I', 'f'], [1, 0, 1], [1.0, 0.0, 0.0]]
# [['g', 'h'], [1, 1], [1.0, 1.0]]
#
class BatchIterator(object):
  def __init__(self, size, a, b, vec_a, vec_b):
    self.size = size
    self.current_step = 0
    self.x = [item for x, y in zip(a, b) for item in x + y]
    self.y = [item for x, y in zip(a, b) for item in [vec_a for _ in x] + [vec_b for _ in y]]
    self.z = self._state_clearers()
    self._crop()
    self.length = len(self.x)

  def __iter__(self):
    return self

  def __next__(self):
    current_index = self.current_step * self.size

    if current_index >= self.length:
      raise StopIteration
    else:
      self.current_step += 1
      return [self.x[current_index:current_index + self.size],
              self.y[current_index:current_index + self.size],
              self.z[current_index:current_index + self.size]]

  # this method rewinds the iterator to make an infinite loop
  def next(self):
    try:
      result = self.__next__()
    except StopIteration:
      self.current_step = 0
      result = self.__next__()

    return result

  def rewind(self):
    self.current_step = 0

  def _state_clearers(self):
    modifiers = []
    previous_label = self.y[0]

    for label in self.y:
      if label == previous_label:
        modifier = 1.
      else:
        modifier = 0.

      previous_label = label
      modifiers.append(modifier)

    return modifiers

  def _crop(self):
    crop_index = len(self.x) % self.size
    if crop_index == 0: return

    for array in [self.x, self.y, self.z]:
      array[:] = array[:-crop_index]
