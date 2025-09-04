"""XYZ parsing and related helpers (moved from base.aux)."""
from __future__ import annotations
import numpy as np, yaml

def xyz2np(xyz: str):
    xyz = xyz.splitlines()
    num_of_atoms = int(xyz[0])
    ans = np.zeros((num_of_atoms, 3))
    j = 0; atoms = []; unique_labels = dict()
    for line in xyz[2:]:
        if len(line.strip()) > 0:
            temp = line.split(); label = ''.join([i for i in temp[0] if not i.isdigit()])
            try:
                unique_labels[label] += 1; temp[0] = label + str(unique_labels[label])
            except KeyError:
                temp[0] = label + '1'; unique_labels[label] = 1
            atoms.append(temp[0])
            ans[j, 0] = float(temp[1]); ans[j, 1] = float(temp[2]); ans[j, 2] = float(temp[3]); j += 1
    return atoms, ans

def dict2xyz(input_data):
    if not isinstance(input_data, dict):
        return input_data
    output = str(input_data['num_atoms']) + '\n' + str(input_data['title']) + '\n'
    for j in range(input_data['num_atoms']):
        output += list(input_data['atoms'][j].keys())[0] + "    " + \
                  str(list(input_data['atoms'][j].values())[0][0]) + "    " + \
                  str(list(input_data['atoms'][j].values())[0][1]) + "    " + \
                  str(list(input_data['atoms'][j].values())[0][2]) + "\n"
    return output

def yaml_parser(input_data):
    output = None
    if isinstance(input_data,str) and input_data.lower().endswith(('.yml', '.yaml')):
        with open(input_data, 'r') as stream:
            output = yaml.safe_load(stream)
    else:
        try:
            output = yaml.safe_load(input_data)
        except yaml.YAMLError:
            pass
    if isinstance(output, dict) and 'primitive_cell' in output and 'lattice_constant' in output:
        import numpy as np
        output['primitive_cell'] = np.array(output['primitive_cell']) * output['lattice_constant']
    return output
