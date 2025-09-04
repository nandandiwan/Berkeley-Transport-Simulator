import numpy as np
from itertools import product
from .aux import accum

def cut_in_blocks(h_0, blocks):
    j1 = 0; h_0_s=[]; h_l_s=[]; h_r_s=[]
    for j, block in enumerate(blocks):
        h_0_s.append(h_0[j1:block + j1, j1:block + j1])
        if j < len(blocks) - 1:
            h_l_s.append(h_0[block + j1:block + j1 + blocks[j + 1], j1:block + j1])
            h_r_s.append(h_0[j1:block + j1, j1 + block:j1 + block + blocks[j + 1]])
        j1 += block
    return h_0_s, h_l_s, h_r_s

def compute_edge(mat):
    row, col = np.where(mat != 0.0)
    outeredge = accum(row, col, np.max) + 1; outeredge[0] = max(0, outeredge[0]); outeredge = np.maximum.accumulate(outeredge)
    outeredge1 = accum(np.max(row) - row[::-1], np.max(row) - col[::-1], np.max) + 1
    outeredge1[0] = max(0, outeredge1[0]); outeredge1 = np.maximum.accumulate(outeredge1)
    return outeredge, outeredge1

def blocksandborders_constrained(left_block, right_block, edge, edge1):
    size = len(edge); left_block = max(1, left_block); right_block = max(1, right_block)
    if left_block + right_block < size:
        new_left_block = edge[left_block - 1] - left_block
        new_right_block = edge1[right_block - 1] - right_block
        if left_block + new_left_block <= size - right_block and size - right_block - new_right_block >= left_block:
            blocks = blocksandborders_constrained(new_left_block, new_right_block,
                                                  edge[left_block:-right_block] - left_block,
                                                  edge1[right_block:-left_block] - right_block)
            return [left_block] + blocks + [right_block]
        else:
            if new_left_block > new_right_block:
                return [left_block] + [size - left_block]
            else:
                return [size - right_block] + [right_block]
    elif left_block + right_block == size:
        return [left_block] + [right_block]
    else:
        return [size]

def compute_blocks(left_block, right_block, edge, edge1):
    size = len(edge); ans = blocksandborders_constrained(left_block, right_block, edge, edge1)
    if len(ans) > 2:
        for j, block in enumerate(ans[1:-1]):
            left_block = block; right_block = ans[j + 2]
            ans1 = compute_blocks(left_block, right_block,
                                  edge[sum(ans[:j + 1]):sum(ans[:j + 2])] - sum(ans[:j + 1]),
                                  edge1[sum(ans[:j + 1]):sum(ans[:j + 2])] - sum(ans[:j + 1]))
            ans[j + 1] = ans1
    return ans

def find_nonzero_lines(mat, order):
    if order == 'top':
        line = mat.shape[0]
        while line > 0:
            if np.count_nonzero(mat[line - 1, :]) == 0: line -= 1
            else: break
    elif order == 'bottom':
        line = -1
        while line < mat.shape[0] - 1:
            if np.count_nonzero(mat[line + 1, :]) == 0: line += 1
            else: line = mat.shape[0] - (line + 1); break
    elif order == 'left':
        line = mat.shape[1]
        while line > 0:
            if np.count_nonzero(mat[:, line - 1]) == 0: line -= 1
            else: break
    elif order == 'right':
        line = -1
        while line < mat.shape[1] - 1:
            if np.count_nonzero(mat[:, line + 1]) == 0: line += 1
            else: line = mat.shape[1] - (line + 1); break
    else: raise ValueError('Wrong value of the parameter order')
    return line

def split_into_subblocks(h_0, h_l, h_r):
    if isinstance(h_l, np.ndarray) and isinstance(h_r, np.ndarray):
        h_r_h = find_nonzero_lines(h_r, 'bottom'); h_r_v = find_nonzero_lines(h_r[-h_r_h:, :], 'left')
        h_l_h = find_nonzero_lines(h_l, 'top'); h_l_v = find_nonzero_lines(h_l[:h_l_h, :], 'right')
    if isinstance(h_l, int) and isinstance(h_r, int):
        h_l_h = h_l; h_r_v = h_l; h_r_h = h_r; h_l_v = h_r
    edge, edge1 = compute_edge(h_0)
    left_block = max(h_l_h, h_r_v); right_block = max(h_r_h, h_l_v)
    blocks = blocksandborders_constrained(left_block, right_block, edge, edge1)
    h_0_s, h_l_s, h_r_s = cut_in_blocks(h_0, blocks)
    return h_0_s, h_l_s, h_r_s, blocks

def split_into_subblocks_optimized(h_0, h_l, h_r):
    if isinstance(h_l, np.ndarray) and isinstance(h_r, np.ndarray):
        h_r_h = find_nonzero_lines(h_r, 'bottom'); h_r_v = find_nonzero_lines(h_r[-h_r_h:, :], 'left')
        h_l_h = find_nonzero_lines(h_l, 'top'); h_l_v = find_nonzero_lines(h_l[:h_l_h, :], 'right')
    if isinstance(h_l, int) and isinstance(h_r, int):
        h_l_h = h_l; h_r_v = h_l; h_r_h = h_r; h_l_v = h_r
    edge, edge1 = compute_edge(h_0)
    left_block = max(h_l_h, h_r_v); right_block = max(h_r_h, h_l_v)
    blocks = compute_blocks(left_block, right_block, edge, edge1)
    h_0_s, h_l_s, h_r_s = cut_in_blocks(h_0, blocks)
    return h_0_s, h_l_s, h_r_s, blocks
