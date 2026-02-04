// cppimport


#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include <Eigen/Dense>
#include <vector>
#include <complex>

namespace py = pybind11;
using cdouble = std::complex<double>;
using Matrix = Eigen::MatrixXcd;

struct InverseResult {
    std::vector<Matrix> diagonal;
    std::vector<Matrix> upper;
    std::vector<Matrix> lower;
    std::vector<int> block_sizes;
    Matrix full;
};

static Matrix to_matrix(const py::array& arr, const char* name) {
    if (arr.ndim() != 2 || arr.shape(0) != arr.shape(1)) {
        throw std::runtime_error(std::string(name) + " must be square");
    }
    Eigen::Map<const Matrix> view(reinterpret_cast<const cdouble*>(arr.data()), arr.shape(0), arr.shape(1));
    return Matrix(view);
}

static InverseResult dense_inverse(const std::vector<py::array>& diag_list,
                                   const std::vector<py::array>& upper_list,
                                   const std::vector<py::array>& lower_list,
                                   bool return_full) {
    const std::size_t n_blocks = diag_list.size();
    if (upper_list.size() + 1 != n_blocks || lower_list.size() + 1 != n_blocks) {
        throw std::runtime_error("upper/lower lengths must be n_blocks - 1");
    }
    std::vector<int> sizes;
    sizes.reserve(n_blocks);
    for (std::size_t i = 0; i < n_blocks; ++i) {
        auto m = to_matrix(diag_list[i], "diag");
        sizes.push_back(static_cast<int>(m.rows()));
    }
    std::vector<int> offsets(n_blocks + 1, 0);
    for (std::size_t i = 0; i < n_blocks; ++i) {
        offsets[i + 1] = offsets[i] + sizes[i];
    }
    const int total = offsets.back();
    Matrix A = Matrix::Zero(total, total);
    // place diagonal
    for (std::size_t i = 0; i < n_blocks; ++i) {
        auto block = to_matrix(diag_list[i], "diag");
        const int s = offsets[i];
        A.block(s, s, sizes[i], sizes[i]) = block;
    }
    // place off-diagonals
    for (std::size_t i = 0; i + 1 < n_blocks; ++i) {
        Eigen::Map<const Matrix> upper(reinterpret_cast<const cdouble*>(upper_list[i].data()), sizes[i], sizes[i + 1]);
        Eigen::Map<const Matrix> lower(reinterpret_cast<const cdouble*>(lower_list[i].data()), sizes[i + 1], sizes[i]);
        const int s = offsets[i];
        const int sn = offsets[i + 1];
        A.block(s, sn, sizes[i], sizes[i + 1]) = upper;
        A.block(sn, s, sizes[i + 1], sizes[i]) = lower;
    }
    // invert
    Matrix inv = A.inverse();

    InverseResult res;
    res.block_sizes = sizes;
    res.diagonal.reserve(n_blocks);
    res.upper.reserve(n_blocks > 0 ? n_blocks - 1 : 0);
    res.lower.reserve(n_blocks > 0 ? n_blocks - 1 : 0);

    for (std::size_t i = 0; i < n_blocks; ++i) {
        const int s = offsets[i];
        const int e = offsets[i + 1] - s;
        res.diagonal.emplace_back(inv.block(s, s, e, e));
        if (i + 1 < n_blocks) {
            const int sn = offsets[i + 1];
            const int en = offsets[i + 2] - sn;
            res.upper.emplace_back(inv.block(s, sn, e, en));
            res.lower.emplace_back(inv.block(sn, s, en, e));
        }
    }
    if (return_full) {
        res.full = inv;
    }
    return res;
}

py::object pairwise_inverse_cpp(py::list diag_list, py::list upper_list, py::list lower_list, bool return_full) {
    std::vector<py::array> diag(diag_list.size());
    std::vector<py::array> upper(upper_list.size());
    std::vector<py::array> lower(lower_list.size());
    for (std::size_t i = 0; i < diag.size(); ++i) diag[i] = py::cast<py::array>(diag_list[i]);
    for (std::size_t i = 0; i < upper.size(); ++i) upper[i] = py::cast<py::array>(upper_list[i]);
    for (std::size_t i = 0; i < lower.size(); ++i) lower[i] = py::cast<py::array>(lower_list[i]);

    auto res = dense_inverse(diag, upper, lower, return_full);

    py::list diag_out;
    py::list upper_out;
    py::list lower_out;
    for (const auto& m : res.diagonal) diag_out.append(py::cast(m));
    for (const auto& m : res.upper) upper_out.append(py::cast(m));
    for (const auto& m : res.lower) lower_out.append(py::cast(m));

    py::tuple block_sizes(res.block_sizes.size());
    for (std::size_t i = 0; i < res.block_sizes.size(); ++i) block_sizes[i] = res.block_sizes[i];

    if (return_full) {
        return py::make_tuple(diag_out, upper_out, lower_out, block_sizes, py::cast(res.full));
    }
    return py::make_tuple(diag_out, upper_out, lower_out, block_sizes, py::none());
}

PYBIND11_MODULE(pairwise_partial_inverse_ext, m) {
    m.doc() = "C++ dense inversion for block-tridiagonal matrices";
    m.def("pairwise_inverse_cpp", &pairwise_inverse_cpp, py::arg("diagonal_blocks"), py::arg("upper_blocks"), py::arg("lower_blocks"), py::arg("return_full") = false);
}
