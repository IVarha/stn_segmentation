#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pySurface.h>
#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

int add(int i, int j) {
    return i + j;
}

namespace py = pybind11;

PYBIND11_MODULE(ExtPy, m) {
m.doc() = R"pbdoc(
        Pybind11 example plugin
        -----------------------
        .. currentmodule:: cmake_example
        .. autosummary::
           :toctree: _generate
           add
           subtract
    )pbdoc";

m.def("add", &add, R"pbdoc(
        Add two numbers
        Some other explanation about the add function.
    )pbdoc");

m.def("subtract", [](int i, int j) { return i - j; }, R"pbdoc(
        Subtract two numbers
        Some other explanation about the subtract function.
    )pbdoc");


    py::class_<pySurface>(m, "cMesh")
            .def(py::init<const std::string &>())
            .def("getName",&pySurface::getName)
            .def("setNewPoints", &pySurface::modify_points)
            .def("selfIntersectionTest", &pySurface::self_intersection_test)
            .def("applyTransformation", &pySurface::apply_transformation);

#ifdef VERSION_INFO
m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
m.attr("__version__") = "dev";
#endif
}
