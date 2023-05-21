from abc import ABC
from abc import abstractmethod


class Sequence(ABC):
    # Container
    @abstractmethod
    def build(self, x):
        pass

    @abstractmethod
    def len(self):
        pass

    # Static
    @abstractmethod
    def iter_seq(self):
        pass

    @abstractmethod
    def get_at(self, i):
        pass

    @abstractmethod
    def set_at(self, i, x):
        pass

    # Dynamic
    @abstractmethod
    def insert_at(self, i, x):
        pass

    @abstractmethod
    def delete_at(self, i):
        pass

    @abstractmethod
    def insert_first(self, x):
        pass

    @abstractmethod
    def delete_first(self):
        pass

    @abstractmethod
    def insert_last(self, x):
        pass

    @abstractmethod
    def delete_last(self):
        pass


class Set(ABC):
    # Container
    @abstractmethod
    def build(self, x):
        pass

    @abstractmethod
    def len(self):
        pass

    # Static
    @abstractmethod
    def find(self, k):
        pass

    # Dynamic
    @abstractmethod
    def insert(self, x):
        pass

    @abstractmethod
    def delete(self, k):
        pass

    # Order
    @abstractmethod
    def iter_ord(self):
        pass

    @abstractmethod
    def find_min(self):
        pass

    @abstractmethod
    def find_max(self):
        pass

    @abstractmethod
    def find_next(self, k):
        pass

    @abstractmethod
    def find_prev(self, k):
        pass
