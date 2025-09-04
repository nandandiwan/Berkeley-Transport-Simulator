from __future__ import print_function, absolute_import
import numpy as np
from itertools import product

def accum(accmap, input, func=None, size=None, fill_value=0, dtype=None):
    if accmap.shape[:input.ndim] != input.shape:
        raise ValueError("The initial dimensions of accmap must be the same as a.shape")
    if func is None:
        func = np.sum
    if dtype is None:
        dtype = input.dtype
    if accmap.shape == input.shape:
        accmap = np.expand_dims(accmap, -1)
    adims = tuple(range(input.ndim))
    if size is None:
        size = 1 + np.squeeze(np.apply_over_axes(np.max, accmap, axes=adims))
    size = np.atleast_1d(size)
    vals = np.empty(size, dtype='O')
    for s in product(*[range(k) for k in size]):
        vals[s] = []
    for s in product(*[range(k) for k in input.shape]):
        indx = tuple(accmap[s]); vals[indx].append(input[s])
    out = np.empty(size, dtype=dtype)
    for s in product(*[range(k) for k in size]):
        out[s] = fill_value if vals[s] == [] else func(vals[s])
    return out

def count_species(list_of_labels):
    counter = {}
    for item in list_of_labels:
        key = ''.join([i for i in item if not i.isdigit()])
        counter[key] = counter.get(key, 0) + 1
    return counter

def is_in_coords(coords, coords_in):
    for item in coords_in:
        if np.linalg.norm(np.array(coords) - np.array(item)) < 1e-6:
            return True
    return False

def print_dict(dictionary):
    out = "{:<18} {:<15} \n".format('Label', 'Coordinates')
    for key, value in dictionary.items():
        out += "{:<18} {:<15} \n".format(key, str(value))
    return out

def print_table(myDict, colList=None, sep='\uFFFA'):
    import numpy as _np
    if not colList:
        colList = list(myDict[0].keys() if myDict else [])
    myList = [colList]
    for item in myDict:
        myList.append([str(item[col]) for col in colList])
    colSize = [max(map(len, (sep.join(col)).split(sep))) for col in zip(*myList)]
    formatStr = ' | '.join(["{{:<{}}}".format(i) for i in colSize])
    line = formatStr.replace(' | ', '-+-').format(*['-' * i for i in colSize])
    item = myList.pop(0); lineDone = False; out = "\n"
    while myList:
        if all(not i for i in item):
            item = myList.pop(0)
            if line and (sep != '\uFFFA' or not lineDone):
                out += line + "\n"; lineDone = True
        row = [i.split(sep, 1) for i in item]
        out += formatStr.format(*[i[0] for i in row]) + "\n"
        item = [i[1] if len(i) > 1 else '' for i in row]
    out += line + "\n"; return out
