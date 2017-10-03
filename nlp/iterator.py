# Usage example
# =============
#
# batch_size = 3
# a = [['A', 'B', 'C', 'D'], ['E', 'F'], ['G'], ['H', 'I', 'J']]
# b = [['a', 'b', 'c'], ['d'], ['e', 'f']]
#
# print(f'Good articles({len(a)}): {a}')
# print(f'Bad  articles({len(b)}): {b}')
#
# batches = BatchIterator(batch_size, a, b, 0, 1)
#
# print(f'\nGenerated batches of length {batch_size}:')
# for batch in batches:
#   print(batch)
#
#
# Output
# ======
#
# Good articles(4): [['A', 'B', 'C', 'D'], ['E', 'F'], ['G'], ['H', 'I', 'J']]
# Bad  articles(3): [['a', 'b', 'c'], ['d'], ['e', 'f']]
#
# Generated batches of length 3:
# [['A', 'B', 'C'], [0, 0, 0]]
# [['D', 'a', 'b'], [0, 1, 1]]
# [['c', 'E', 'F'], [1, 0, 0]]
# [['d', 'G', 'e'], [1, 0, 1]]
# [['f'], [1]]
#
class BatchIterator(object):
  def __init__(self, size, a, b, vec_a, vec_b):
    self.size = size
    self.current = 0
    self.x = [item for x, y in zip(a, b) for item in x + y]
    self.y = [item for x, y in zip(a, b) for item in [vec_a for _ in x] + [vec_b for _ in y]]
    self.length = len(self.x)

  def __iter__(self):
    return self

  def __next__(self):
    if self.current * self.size > self.length:
      raise StopIteration
    else:
      current_index = self.current * self.size
      self.current += 1
      return [self.x[current_index:current_index + self.size],
              self.y[current_index:current_index + self.size]]

  # this method rewinds the iterator to make an infinite loop
  def next(self):
    try:
      result = self.__next__()
    except StopIteration:
      self.current = 0
      result = self.__next__()

    return result
