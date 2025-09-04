"""Auxiliary functions (verbatim copy needed for identical Hamiltonian generation)."""
from __future__ import print_function
from __future__ import absolute_import
from itertools import product
import numpy as np
import yaml


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


def xyz2np(xyz):
    xyz = xyz.splitlines()
    num_of_atoms = int(xyz[0])
    ans = np.zeros((num_of_atoms, 3))
    j = 0
    atoms = []
    unique_labels = dict()
    for line in xyz[2:]:
        if len(line.strip()) > 0:
            temp = line.split()
            label = ''.join([i for i in temp[0] if not i.isdigit()])
            try:
                unique_labels[label] += 1
                temp[0] = label + str(unique_labels[label])
            except KeyError:
                temp[0] = label + '1'
                unique_labels[label] = 1
            atoms.append(temp[0])
            ans[j, 0] = float(temp[1]); ans[j, 1] = float(temp[2]); ans[j, 2] = float(temp[3])
            j += 1
    return atoms, ans


def count_species(list_of_labels):
    counter = {}
    for item in list_of_labels:
        key = ''.join([i for i in item if not i.isdigit()])
        try:
            counter[key] += 1
        except KeyError:
            counter[key] = 1
    return counter


def dict2xyz(input_data):
    if not isinstance(input_data, dict):
        return input_data
    output = str(input_data['num_atoms']) + '\n'
    output += str(input_data['title']) + '\n'
    for j in range(input_data['num_atoms']):
        output += list(input_data['atoms'][j].keys())[0] + \
                  "    " + str(list(input_data['atoms'][j].values())[0][0]) + \
                  "    " + str(list(input_data['atoms'][j].values())[0][1]) + \
                  "    " + str(list(input_data['atoms'][j].values())[0][2]) + "\n"
    return output


def yaml_parser(input_data):
    output = None
    if isinstance(input_data,str) and input_data.lower().endswith(('.yml', '.yaml')):
        with open(input_data, 'r') as stream:
            try:
                output = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)
    else:
        try:
            output = yaml.safe_load(input_data)
        except yaml.YAMLError as exc:
            print(exc)
    if isinstance(output, dict) and 'primitive_cell' in output and 'lattice_constant' in output:
        output['primitive_cell'] = np.array(output['primitive_cell']) * output['lattice_constant']
    return output


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
    if not colList:
        colList = list(myDict[0].keys() if myDict else [])
    myList = [colList]
    for item in myDict:
        myList.append([str(item[col]) for col in colList])
    colSize = [max(map(len, (sep.join(col)).split(sep))) for col in zip(*myList)]
    formatStr = ' | '.join(["{{:<{}}}".format(i) for i in colSize])
    line = formatStr.replace(' | ', '-+-').format(*['-' * i for i in colSize])
    item = myList.pop(0)
    lineDone = False
    out = "\n"
    while myList:
        if all(not i for i in item):
            item = myList.pop(0)
            if line and (sep != '\uFFFA' or not lineDone):
                out += line + "\n"; lineDone = True
        row = [i.split(sep, 1) for i in item]
        out += formatStr.format(*[i[0] for i in row]) + "\n"
        item = [i[1] if len(i) > 1 else '' for i in row]
    out += line + "\n"
    return out
