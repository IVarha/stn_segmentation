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


m.def( "is_triangle_intersected",&pySurface::triangles_intersected,R"pbdoc(
        Test if triangles formed by points intersected
        Some other explanation about the subtract function.
    )pbdoc");

    py::class_<pySurface>(m, "cMesh")
            .def(py::init<const std::string &>())
            .def("getName",&pySurface::getName)
            .def("modify_points", &pySurface::modify_points)
            .def("selfIntersectionTest", &pySurface::self_intersection_test)
            .def("apply_transform", &pySurface::apply_transformation)
            .def("generate_normals", &pySurface::generateNormals)
            .def("generate_mesh_points", &pySurface::getInsideMeshPoints);

#ifdef VERSION_INFO
m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
m.attr("__version__") = "dev";
#endif
}
