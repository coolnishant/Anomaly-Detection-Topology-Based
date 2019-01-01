import bz2
import csv
import functools
import gzip
import os
from functools import wraps
from builtins import str
from six import iteritems


class memoize(object):
    """Decorator. Caches a function's return value each time it is called.
    If called later with the same arguments, the cached value is returned
    (not reevaluated).
    """

    def __init__(self, func):
        self.func = func
        self.cache = {}

    def __call__(self, *args):
        if args in self.cache:
            return self.cache[args]
        else:
            value = self.func(*args)
            self.cache[args] = value
            return value

    def __repr__(self):
        """Return the function's docstring."""
        return self.func.__doc__

    def __get__(self, obj, objtype):
        """Support instance methods."""
        fn = functools.partial(self.__call__, obj)
        fn.reset = self._reset

        return fn

    def _reset(self):
        self.cache = {}


def memoize2(f):
    cache = {}

    @wraps(f)
    def inner(obj, arg1, arg2):
        memo_key = str(arg1) + str(arg2)
        if memo_key not in cache:
            cache[memo_key] = f(obj, arg1, arg2)
        return cache[memo_key]

    return inner


def read_file_by_lines(path):
    with open(path, "r") as f:
        line_list = f.read().splitlines()
    return line_list


def read_file(path):
    with open(path, "r") as f:
        for line in f:
            yield line


def read_bz2(path):
    with bz2.BZ2File(path, "rt") as f:
        for line in f:
            yield line


def read_gzip(path):
    with gzip.open(path, "rt") as f:
        for line in f:
            yield line


def append_to_file(data, path):
    with open(path, 'a') as f:
        f.write(data)
        f.close()

def union(*args):
    """ return the union of two lists """
    # return list(set(a) | set(b))
    return set().union(*args)


def intersect(a, b):
    """ return the intersection of two lists """
    return set(a) & set(b)


def extract_items_from_line(line, delimiter=","):
    return [item.strip() for item in line.rstrip('\r\n').replace('"', '').split(delimiter)]


def write_to_file(path, data):
    with open(path, 'w') as f:
        f.write(data)
        f.close()


# Convert List to string
def list_to_string(str_list, delimiter=","):
    return delimiter.join(map(str, str_list))


def two_dimensional_list_to_string(two_dim_list):
    str_list = []
    for row in two_dim_list:
        str_list.append(list_to_string(row))
    return list_to_string(str_list, "\n")


def to_iterable(item):
    if item is None:  # include all nodes via iterator
        item = []
    elif not hasattr(item, "__iter__") or isinstance(item, str):  # if vertices is a single node
        item = [item]  # ?iter()
    return item


def dict_writer(mydict, output_path, mode='wb'):
    is_new_file = False
    if not is_valid_path(output_path):
        open(output_path, "w").close()
        is_new_file = True
    with open(output_path, mode) as f:  # Just use 'w' mode in 3.x
        writer = csv.DictWriter(f, mydict[0].keys(), lineterminator='\n')
        if is_new_file:
            writer.writeheader()
        writer.writerows(mydict)


def is_valid_path(path):
    if (isinstance(path, str) or isinstance(path, str)) and os.path.exists(path):
        return True
    return False


def write_hash_table(hash_table, output_path, header=None):
    # csv_list = [[key, value] for key, value in hash_table.iteritems()]
    csv_list = list(iteritems(hash_table))
    if header is not None:
        csv_list = [header] + csv_list
    write_to_file(output_path, two_dimensional_list_to_string(csv_list))


def delete_file_content(file_path):
    if is_valid_path(file_path):
        os.remove(file_path)
