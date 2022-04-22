from itertools import islice, tee


# wrapper for python dictionaries
class DotDict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


# dictionary conversions
dl_to_ld = lambda dl: [dict(zip(dl, t)) for t in zip(*dl.values())]
ld_to_dl = lambda ld: (lambda l: {k: [d[k] for d in l] for k in l[0]})(list(ld))
key_map = lambda map_func, dic: {map_func(k): v for k, v in dic.items()}
val_map = lambda map_func, dic: {k: map_func(v) for k, v in dic.items()}


# get pairs from iterator, with variable step size
def pairwise(iterable, step=1):
    a, b = tee(iterable)
    next(b, None)
    if step != 1:
        a, b = [islice(x, None, None, step) for x in [a, b]]
    return zip(a, b)


# check if an iterable only contains one value and get it
def check_single_value(iterable, skip_none=False):
    i = (v for v in iterable if v is not None) if skip_none else iter(iterable)
    v = next(i)
    assert not any(True for _ in i), 'iterable must not contain more than one value'
    return v