#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include "linear_regression.h"
#include "lossFunctions.h"
#include "pca.h"
#include "eigen.h"

namespace py = pybind11;

PYBIND11_MODULE(metnum, m) {
    py::class_<LinearRegression>(m, "LinearRegression")
        .def(py::init<>())
        .def("fit", &LinearRegression::fit)
        .def("fitWithSVD", &LinearRegression::fitWithSVD)
        .def("fitWithQR", &LinearRegression::fitWithQR)
        .def("fitWithNormalEq", &LinearRegression::fitWithNormalEq)
        .def("predict", &LinearRegression::predict);
    py::class_<LossFunctions>(m, "LossFunctions")
        .def(py::init<>())
        .def("meanAbsoluteError", &LossFunctions::meanAbsoluteError)
        .def("meanSquareError", &LossFunctions::meanSquareError)
        .def("rootMeanSquareError", &LossFunctions::rootMeanSquareError)
        .def("rootMeanSquareLogError", &LossFunctions::rootMeanSquareLogError);
    py::class_<PCA>(m, "PCA")
        .def(py::init<unsigned int>())
        .def("fit", &PCA::fit)
        .def("transform", &PCA::transform)
        .def("getEigenvectorMatrix", &PCA::getEigenvectorMatrix)
        .def("getEigenvaluesMatrix", &PCA::getEigenvaluesMatrix);
    m.def(
        "power_iteration", &power_iteration,
        "Function that calculates eigenvector",
        py::arg("X"),
        py::arg("num_iter")=5000,
        py::arg("epsilon")=1e-16
    );
    m.def(
        "get_first_eigenvalues", &get_first_eigenvalues,
        "Function that calculates eigenvector",
        py::arg("X"),
        py::arg("num"),
        py::arg("num_iter")=5000,
        py::arg("epsilon")=1e-16
    );
}
