#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pySurface.h>
//#include <pyNiftiImage.h>
#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

int add(int i, int j) {
    return i + j;
}


std::vector<std::vector<double>> apply_transform_2_pts(std::vector<std::vector<double>> norms, std::vector<std::vector<double>> matr){
    auto res = std::vector<std::vector<double>>();
    for (auto & norm : norms){
        auto tmp  = std::vector<double>(3);
        tmp[0] = matr[0][0]* norm[0] + matr[0][1]* norm[1] + matr[0][2]* norm[2] + matr[0][3];
        tmp[1] = matr[1][0]* norm[0] + matr[1][1]* norm[1] + matr[1][2]* norm[2] + matr[1][3];
        tmp[2] = matr[2][0]* norm[0] + matr[2][1]* norm[1] + matr[2][2]* norm[2] + matr[2][3];
        res.push_back(tmp);
    }
    return res;

}

std::vector<std::vector<std::vector<double>>> apply_transform_2_norms(std::vector<std::vector<std::vector<double>>> norms, std::vector<std::vector<double>> matr){
    auto res = std::vector<std::vector<std::vector<double>>>();
    int vsize = norms[0].size();
    for (auto & norm : norms){
        auto tmpm = std::vector<std::vector<double>>();
        for (int j = 0; j< vsize;j++){
            auto tmp  = std::vector<double>(3);
            tmp[0] = matr[0][0]* norm[j][0] + matr[0][1]* norm[j][1] + matr[0][2]* norm[j][2] + matr[0][3];
            tmp[1] = matr[1][0]* norm[j][0] + matr[1][1]* norm[j][1] + matr[1][2]* norm[j][2] + matr[1][3];
            tmp[2] = matr[2][0]* norm[j][0] + matr[2][1]* norm[j][1] + matr[2][2]* norm[j][2] + matr[2][3];
            tmpm.push_back(tmp);

        }
        res.push_back(tmpm);

    }
    return res;

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

    m.def("apply_transform_2_pts", &apply_transform_2_pts, R"pbdoc(
        Applies transform to all points
        Some other explanation about the subtract function.
    )pbdoc");

    m.def("apply_transform_2_norms", &apply_transform_2_norms, R"pbdoc(
        Applies transform to pts in normals
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
            .def("save_obj",&pySurface::saveObj)
            .def("generate_mesh_points", &pySurface::getInsideMeshPoints)
            .def("get_unpacked_coords",&pySurface::getUnpackedCords)
            .def("calculate_volume",&pySurface::getUnpackedCords);


//    py::class_<pyNiftiImage>(m,"cImage")
//            .def(py::init<std::string>())
//            .def("loadMask", &pyNiftiImage::setMask)
//            .def("interpolate_normals", &pyNiftiImage::interpolate_normals);

#ifdef VERSION_INFO
m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
m.attr("__version__") = "dev";
#endif
}
