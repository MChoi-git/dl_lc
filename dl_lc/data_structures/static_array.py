from interfaces import Sequence, Set


class StaticArray(Sequence):
    def __init__(self):
        self.elements = []

    def build(self, x):
        for e in x:
            self.elements.append(e)

    def len(self):
        length = 0
        for _ in self.elements:
            length += 1
        return length

    def iter_seq(self):
        return iter(self.elements)

    def get_at(self, i):
        return self.elements[i]

    def set_at(self, i, x):
        self.elements[i] = x

    def insert_at(self, i, x):
        # Handle empty array.
        if self.len() == 0:
            self.elements.append(x)

        # Handle out of bounds.
        elif i >= self.len():
            raise IndexError

        # Handle general case.
        else:
            # Store elements to shift right.
            tmp = self.elements[i:]

            # Set new element.
            self.elements[i] = x
            
            # Copy elements after index back into buffer.
            # NOTE: Python lists are dynamic, but regularly O(n) to reallocate
            #       buffer.
            self.elements[i + 1:] = tmp

    def delete_at(self, i):
        # Handle out of bounds.
        if i >= self.len():
            raise IndexError

        # Handle empty array.
        elif self.len() == 0:
            pass

        # Handle general case.
        else:
            # Save elements after index.
            tmp = self.elements[i + 1:]
            
            # Shift elements left.
            self.elements[i:] = tmp

    def insert_first(self, x):
        self.insert_at(0, x)

    def delete_first(self):
        self.delete_at(0)

    def insert_last(self, x):
        self.elements.append(x)

    def delete_last(self):
        del self.elements[-1]


# Testing
# TODO: Move into interface since tests can be shared among children.
x = StaticArray()
x.build([1, 2, 3, 4, 5])
assert x.elements == [1, 2, 3, 4, 5]
assert x.len() == 5

assert x.get_at(0) == 1
x.set_at(0, 6)
assert x.elements[0] == 6

x.insert_at(0, -1)
assert x.elements == [-1, 6, 2, 3, 4, 5]

x.delete_at(0)
assert x.elements == [6, 2, 3, 4, 5]

x.delete_at(4)
assert x.elements == [6, 2, 3, 4]

x.insert_first(-1)
assert x.elements == [-1, 6, 2, 3, 4]

x.elements = []
x.insert_first(0)
x.insert_first(1)
assert x.elements == [1, 0]

x.delete_first()
assert x.elements == [0]
x.delete_first()
assert x.elements == []

x.insert_last(0)
assert x.elements == [0]
x.delete_last()
assert x.elements == []
