import copy
import numpy as np
import scipy.linalg as linalg

def mat_left_div(mat_a, mat_b):
    ans, resid, rank, s = linalg.lstsq(mat_a, mat_b, lapack_driver='gelsy')
    return ans

def _recursive_gf(energy, mat_l_list, mat_d_list, mat_u_list, s_in=0, s_out=0, damp=0.000001j):
    for jj,item in enumerate(mat_d_list):
        mat_d_list[jj]=item - np.diag(energy*np.ones(mat_d_list[jj].shape[0]) + 1j*damp)
    num_of_matrices=len(mat_d_list)
    mat_shapes=[item.shape for item in mat_d_list]
    gr_left=[None for _ in range(num_of_matrices)]
    gr_left[0]=mat_left_div(-mat_d_list[0], np.eye(mat_shapes[0][0]))
    for q in range(num_of_matrices-1):
        gr_left[q+1]=mat_left_div((-mat_d_list[q+1]-mat_l_list[q].dot(gr_left[q]).dot(mat_u_list[q])), np.eye(mat_shapes[q+1][0]))
    grl=[None for _ in range(num_of_matrices-1)]
    gru=[None for _ in range(num_of_matrices-1)]
    grd=copy.copy(gr_left)
    g_trans=copy.copy(gr_left[len(gr_left)-1])
    for q in range(num_of_matrices-2,-1,-1):
        grl[q]=grd[q+1].dot(mat_l_list[q]).dot(gr_left[q])
        gru[q]=gr_left[q].dot(mat_u_list[q]).dot(grd[q+1])
        grd[q]=gr_left[q]+gr_left[q].dot(mat_u_list[q]).dot(grl[q])
        g_trans=gr_left[q].dot(mat_u_list[q]).dot(g_trans)
    for jj,item in enumerate(mat_d_list):
        mat_d_list[jj]=mat_d_list[jj]+np.diag(energy*np.ones(mat_d_list[jj].shape[0]) + 1j*damp)
    return g_trans, grd, grl, gru, gr_left

def recursive_gf(energy, mat_l_list, mat_d_list, mat_u_list, left_se=None, right_se=None, s_in=0, s_out=0, damp=0.000001j):
    if isinstance(left_se, np.ndarray):
        s01,s02=mat_d_list[0].shape; left_se=left_se[:s01,:s02]; mat_d_list[0]=mat_d_list[0]+left_se
    if isinstance(right_se, np.ndarray):
        s11,s12=mat_d_list[-1].shape; right_se=right_se[-s11:,-s12:]; mat_d_list[-1]=mat_d_list[-1]+right_se
    ans=_recursive_gf(energy, mat_l_list, mat_d_list, mat_u_list, s_in=s_in, s_out=s_out, damp=damp)
    if isinstance(left_se, np.ndarray):
        mat_d_list[0]=mat_d_list[0]-left_se
    if isinstance(right_se, np.ndarray):
        mat_d_list[-1]=mat_d_list[-1]-right_se
    return ans
