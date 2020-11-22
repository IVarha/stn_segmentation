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

    void smoothMesh();

private:
    vtkSmartPointer<vtkCellArray> triangles;
    vtkSmartPointer<vtkPolyData> mesh;
    public:
        void read_volume(const std::string& file_name );
        void expand_volume(double mm);
        void write_volume(const std::string file_name);
        void write_obj(const std::string file_name);
        void write_stl(const std::string file_name);
        std::tuple<double, double, double>  centre_of_mesh();
        void shrink_sphere(VolumeDouble& mask, std::tuple<double,double,double> center,double threshold);
        void apply_transformation(TransformMatrix& pre_transformation);

        double calculate_volume();
        static Surface generate_sphere( double radius_mm, std::tuple<double, double, double> centre);

};


#endif //BAYESSIAN_SEGMENTATION_CPP_SURFACE_H
