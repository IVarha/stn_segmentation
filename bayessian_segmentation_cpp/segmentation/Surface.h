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
    vtkSmartPointer<vtkCellArray> triangles;
    vtkSmartPointer<vtkPolyData> mesh;
    static std::shared_ptr<spdlog::logger> _logger;
    //vector of triangles
    std::vector<std::vector<int>> vec_tri;
    //neighbours of triangles;
    std::vector<std::set<int>> tri_neighb;
    // triangles neighboured for each vertice(triangles to which point belong);
    std::vector<std::vector<int>> point_tri;

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

    Surface(const Surface& surface);

    double distanceToPoint(double x, double y, double z);

    void smoothMesh(int iter);

    void saveImage(std::string filename);

    std::vector<std::vector<std::vector<double>>> calculate_normals(double mm, int npts);

    void expand_volume(double mm);
    void write_volume(std::string file_name);
    void write_obj(std::string file_name);
    void write_stl(std::string file_name);
    std::tuple<double, double, double>  centre_of_mesh();
    /**
     * Shrink sphere to label
     * @param mask label of the structure as a mask
     * @param center center of a sphere as [x,y,z]
     * @param threshold value which should be reached for a label considered as a correct
     */
    void shrink_sphere(VolumeDouble& mask, std::tuple<double,double,double> center,double threshold);

    void shrink_sphere(vector<vector<vector<bool>>>& mask, std::tuple<double,double,double> center,double threshold);

    void lab_move_points(VolumeDouble& mask,double threshold,double step);

    void apply_transformation(TransformMatrix& pre_transformation);
    void apply_transformation(const arma::mat& pre_transformation);
    double calculate_volume();

    /**
     * Method generates sphere with radius on a @refitem centre
     * @param radius_mm radius of a generated sphere in mm > 0
     * @param centre centre of a sphere [x,y,z]
     * @param num_of_divisions number of subdivisions number which depends on a number of a vertex (better to use <4)
     * @return Sphere
     */
    static Surface generate_sphere( double radius_mm, std::tuple<double, double, double> centre, int num_of_divisions);

    std::vector<std::vector<double>> rayMeshIntersection(std::vector<std::vector<double>> start_end);

    std::vector<int> rayMeshInterInd(std::vector<std::vector<double>> start_end);

    double* getPoint(int pos);
    /*get Point by id same as previous method
     * */
    Point getPPoint(int id);
    std::vector<std::vector<double>> getTriangleCenters();

private:

    void compute_tri_neigbours();
    void compute_points_neigbours();

    std::vector<Point> get_triangle(int id);

    void update_mesh();
    public:
        void read_volume(const std::string& file_name );



        void read_obj(const string &basicString);

        static bool intersection_triangles(double* v0, double* v1, double* v2, double* v0_2, double* u1, double* u2);


    static bool
    triangle_intersection(const double *V10, const double *V11, const double *V12, const double *V20, const double *V21,
                          const double *V22);

    void lab_move_points_with_stop(VolumeDouble &mask, double threshold, Point center);
};


#endif //BAYESSIAN_SEGMENTATION_CPP_SURFACE_H
