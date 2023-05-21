from abc import ABC


class Sequence(ABC):
    # Container
    @abstractmethod
    def build(x):
        pass

    @abstractmethod
    def len():
        pass

    # Static
    @abstractmethod
    def iter_seq():
        pass

    @abstractmethod
    def get_at(i):
        pass

    @abstractmethod
    def set_at(i):
        pass

    # Dynamic
    @abstractmethod
    def insert_at(i, x):
        pass

    @abstractmethod
    def delete_at(i):
        pass

    @abstractmethod
    def insert_first(x):
        pass

    @abstractmethod
    def delete_first():
        pass

    @abstractmethod
    def insert_last(x):
        pass

    @abstractmethod
    def delete_last():
        pass


class Set(ABC):
    # Container
    @abstractmethod
    def build(x):
        pass

    @abstractmethod
    def len():
        pass

    # Static
    @abstractmethod
    def find(k):
        pass

    # Dynamic
    @abstractmethod
    def insert(x):
        pass

    @abstractmethod
    def delete(k):
        pass

    # Order
    @abstractmethod
    def iter_ord():
        pass

    @abstractmethod
    def find_min():
        pass

    @abstractmethod
    def find_max():
        pass

    @abstractmethod
    def find_next(k):
        pass

    @abstractmethod
    def find_prev(k):
        pass
