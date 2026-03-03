// cppimport


#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <Eigen/Dense>
#include <complex>
#include <vector>
#include <stdexcept>

namespace py = pybind11;
using cdouble = std::complex<double>;
using Matrix = Eigen::MatrixXcd;

static Matrix eye(int n) {
    Matrix I = Matrix::Zero(n, n);
    I.diagonal().setOnes();
    return I;
}

static py::tuple recursive_inverse_cpp(
    double E,
    const Matrix &H,
    const Matrix &Sigma_L_in,
    const Matrix &Sigma_R_in,
    const py::object &overlap_obj,
    double occ_left,
    double occ_right,
    double eta,
    bool compute_lesser,
    bool return_trace
) {
    const int n = static_cast<int>(H.rows());
    if (H.cols() != n) throw std::runtime_error("H must be square");
    const int block_size = static_cast<int>(Sigma_L_in.rows());
    if (Sigma_L_in.cols() != block_size) throw std::runtime_error("Sigma_L must be square");
    if (Sigma_R_in.rows() != Sigma_R_in.cols()) throw std::runtime_error("Sigma_R must be square");
    if (n % block_size != 0) throw std::runtime_error("H dimension must be divisible by Sigma_L dimension (block size)");
    const int n_blocks = n / block_size;

    Matrix S_full;
    if (!overlap_obj.is_none()) {
        S_full = overlap_obj.cast<Matrix>();
        if (S_full.rows() != n || S_full.cols() != n) throw std::runtime_error("overlap_matrix must match H dimensions");
    } else {
        S_full = eye(n);
    }

    const cdouble E_eta = cdouble(E, std::abs(eta));
    Matrix Sigma_L = Sigma_L_in;
    Matrix Sigma_R = Sigma_R_in;
    Matrix Gamma_L = cdouble(0,1) * (Sigma_L - Sigma_L.adjoint());
    Matrix Gamma_R = cdouble(0,1) * (Sigma_R - Sigma_R.adjoint());

    // Build block arrays
    std::vector<Matrix> A_ii;
    std::vector<Matrix> A_ij;
    A_ii.reserve(n_blocks);
    A_ij.reserve(n_blocks > 0 ? n_blocks - 1 : 0);

    for (int i = 0; i < n_blocks; ++i) {
        int s = i * block_size;
        int e = s + block_size;
        Matrix H_block = H.block(s, s, block_size, block_size);
        Matrix S_block = S_full.block(s, s, block_size, block_size);
        Matrix diag_block = E_eta * S_block - H_block;
        if (i == 0) {
            diag_block -= Sigma_L;
        }
        if (i == n_blocks - 1) {
            diag_block -= Sigma_R;
        }
        A_ii.push_back(std::move(diag_block));
        if (i < n_blocks - 1) {
            int sn = e;
            int en = sn + block_size;
            Matrix H_cpl = H.block(s, sn, block_size, block_size);
            Matrix S_cpl = S_full.block(s, sn, block_size, block_size);
            A_ij.push_back(E_eta * S_cpl - H_cpl);
        }
    }

    auto dagger = [](const Matrix &M) { return M.adjoint(); };

    // Forward sweep
    std::vector<Matrix> g_R(n_blocks);
    std::vector<Matrix> g_lesser(n_blocks);

    Matrix A00_eff = A_ii[0];
    g_R[0] = A00_eff.fullPivLu().inverse();
    if (compute_lesser) {
        Matrix Sigma_L_lesser = Gamma_L * occ_left;
        g_lesser[0] = g_R[0] * Sigma_L_lesser * dagger(g_R[0]);
    }

    for (int i = 1; i < n_blocks; ++i) {
        Matrix A_i_im1 = dagger(A_ij[i - 1]);
        Matrix sigma_rec = A_i_im1 * g_R[i - 1] * A_ij[i - 1];
        g_R[i] = (A_ii[i] - sigma_rec).fullPivLu().inverse();
        if (compute_lesser) {
            Matrix sigma_rec_l = A_i_im1 * g_lesser[i - 1] * A_ij[i - 1];
            g_lesser[i] = g_R[i] * sigma_rec_l * dagger(g_R[i]);
        }
    }

    // Backward sweep
    std::vector<Matrix> G_R(n_blocks);
    std::vector<Matrix> G_lesser(n_blocks);
    std::vector<Matrix> G_lesser_offdiag_right(n_blocks > 1 ? n_blocks - 1 : 0);

    if (n_blocks == 1) {
        G_R[0] = g_R[0];
        if (compute_lesser) {
            Matrix Sigma_L_lesser = Gamma_L * occ_left;
            Matrix Sigma_R_lesser = Gamma_R * occ_right;
            Matrix total_sigma_l = Sigma_L_lesser + Sigma_R_lesser;
            G_lesser[0] = G_R[0] * total_sigma_l * dagger(G_R[0]);
        }
    } else {
        Matrix A_N_Nm1 = dagger(A_ij.back());
        Matrix sigma_eff_R = A_N_Nm1 * g_R[n_blocks - 2] * A_ij.back();
        G_R[n_blocks - 1] = (A_ii.back() - sigma_eff_R).fullPivLu().inverse();
        if (compute_lesser) {
            Matrix sigma_eff_l = A_N_Nm1 * g_lesser[n_blocks - 2] * A_ij.back();
            Matrix Sigma_R_lesser = Gamma_R * occ_right;
            Matrix total_sigma_l = Sigma_R_lesser + sigma_eff_l;
            G_lesser[n_blocks - 1] = G_R[n_blocks - 1] * total_sigma_l * dagger(G_R[n_blocks - 1]);
        }
    }

    for (int i = n_blocks - 2; i >= 0; --i) {
        Matrix A_i_ip1 = A_ij[i];
        Matrix A_ip1_i = dagger(A_i_ip1);
        Matrix propagator = g_R[i] * A_i_ip1 * G_R[i + 1] * A_ip1_i;
        G_R[i] = g_R[i] + propagator * g_R[i];
        if (compute_lesser) {
            Matrix gR_dag = dagger(g_R[i]);
            Matrix term1 = g_lesser[i];
            Matrix term2 = propagator * g_lesser[i];
            Matrix term3 = g_lesser[i] * propagator.adjoint();
            Matrix term4 = g_R[i] * A_i_ip1 * G_lesser[i + 1] * A_ip1_i * gR_dag;
            G_lesser[i] = term1 + term2 + term3 + term4;

            Matrix G_ip1_A = dagger(G_R[i + 1]);
            Matrix off_term_R = G_R[i] * A_i_ip1 * G_lesser[i + 1];
            Matrix off_term_L = G_lesser[i] * A_i_ip1 * G_ip1_A;
            G_lesser_offdiag_right[i] = off_term_R + off_term_L;
        }
    }

    // Flatten diagonals
    Matrix G_R_diag(n, 1);
    Matrix G_lesser_diag = Matrix::Zero(n, 1);
    for (int i = 0; i < n_blocks; ++i) {
        int s = i * block_size;
        for (int k = 0; k < block_size; ++k) {
            G_R_diag(s + k, 0) = G_R[i](k, k);
            if (compute_lesser) {
                G_lesser_diag(s + k, 0) = G_lesser[i](k, k);
            }
        }
    }

    cdouble trace_gs = 0.0;
    if (return_trace) {
        for (int i = 0; i < n_blocks; ++i) {
            int s = i * block_size;
            trace_gs += (G_R[i] * S_full.block(s, s, block_size, block_size)).trace();
            if (i < n_blocks - 1) {
                int sn = (i + 1) * block_size;
                trace_gs += (G_R[i] * S_full.block(s, sn, block_size, block_size)).trace();
                trace_gs += (G_R[i + 1] * S_full.block(sn, s, block_size, block_size)).trace();
            }
        }
    }

    py::list offdiag_py;
    for (const auto &m : G_lesser_offdiag_right) offdiag_py.append(py::cast(m));

    return py::make_tuple(
        G_R_diag,
        G_lesser_diag,
        offdiag_py,
        Gamma_L,
        Gamma_R,
        trace_gs
    );
}

PYBIND11_MODULE(recursive_inverse_ext, m) {
    m.doc() = "C++ recursive inverse for block-tridiagonal NEGF";
    m.def("recursive_inverse_cpp", &recursive_inverse_cpp,
          py::arg("E"),
          py::arg("H"),
          py::arg("Sigma_L"),
          py::arg("Sigma_R"),
          py::arg("overlap_matrix"),
            py::arg("occ_left") = 0.0,
            py::arg("occ_right") = 0.0,
          py::arg("eta") = 1e-6,
          py::arg("compute_lesser") = true,
          py::arg("return_trace") = false);
}
