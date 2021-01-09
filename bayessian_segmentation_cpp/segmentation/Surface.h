//
// Created by ivarh on 09/11/2020.
//

#ifndef BAYESSIAN_SEGMENTATION_CPP_SURFACE_H
#define BAYESSIAN_SEGMENTATION_CPP_SURFACE_H

#include <string>
#include <list>
#include <vtkPoints.h>
#include <vtkCellArray.h>
#include <vtkPolyData.h>
#include "vtkSmartPointer.h"
#include <NiftiImage.h>
#include "spdlog/spdlog.h"
#include "spdlog/logger.h"
#include "spdlog/sinks/rotating_file_sink.h"
#include <set>

class Surface {

    private:

    vtkSmartPointer<vtkPoints> points;
public:
    const vtkSmartPointer<vtkPoints> &getPoints() const;

    void setPoints(const vtkSmartPointer<vtkPoints> &points);

    const vtkSmartPointer<vtkCellArray> &getTriangles() const;

    void setTriangles(const vtkSmartPointer<vtkCellArray> &triangles);

    const vtkSmartPointer<vtkPolyData> &getMesh() const;

    void setMesh(const vtkSmartPointer<vtkPolyData> &mesh);

    void apply_points(std::vector<double>& new_pts);

    std::vector<std::vector<int>> getTrianglesAsVec();

    std::vector<std::vector<double>> getPointsAsVec();

    void triangle_normalisation(int iterations,double fraction);

    Surface();

    void smoothMesh();

    void expand_volume(double mm);
    void write_volume(const std::string file_name);
    void write_obj(const std::string file_name);
    void write_stl(const std::string file_name);
    std::tuple<double, double, double>  centre_of_mesh();
    void shrink_sphere(VolumeDouble& mask, std::tuple<double,double,double> center,double threshold);
    void lab_move_points(VolumeDouble& mask,double threshold );

    void apply_transformation(TransformMatrix& pre_transformation);
    void apply_transformation(arma::mat pre_transformation);
    double calculate_volume();
    static Surface generate_sphere( double radius_mm, std::tuple<double, double, double> centre);


    double* getPoint(int pos);


private:
    vtkSmartPointer<vtkCellArray> triangles;
    vtkSmartPointer<vtkPolyData> mesh;
    static std::shared_ptr<spdlog::logger> _logger;
    //vector of triangles
    std::vector<std::vector<int>> vec_tri;
    //neighbours of triangles;
    std::vector<std::set<int>> tri_neighb;
    // triangles neighboured for each vertice(triangles to which point belong);
    std::vector<std::vector<int>> point_tri;
    void compute_tri_neigbours();
    void compute_points_neigbours();

    void update_mesh();
    public:
        void read_volume(const std::string& file_name );



        void read_obj(const string &basicString);

        static bool intersection_triangles(double* v0, double* v1, double* v2, double* v0_2, double* u1, double* u2);


    static bool
    triangle_intersection(const double *V10, const double *V11, const double *V12, const double *V20, const double *V21,
                          const double *V22);
};


#endif //BAYESSIAN_SEGMENTATION_CPP_SURFACE_H
