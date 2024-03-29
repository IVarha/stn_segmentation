//
// Created by ivarh on 11/12/2020.
//

#ifndef BAYESSIAN_SEGMENTATION_CPP_PYSURFACE_H
#define BAYESSIAN_SEGMENTATION_CPP_PYSURFACE_H
#include <string>
#include "Surface.h"
#include "iostream"
#include "triangleintersects.h"
#include <set>
class pySurface {
    std::string name;
    std::vector<std::vector<int>> triangles;
    std::vector<std::set<int>> neighb_tri;

    vtkSmartPointer<vtkPoints> points;
    Surface* mesh = nullptr;
    int p_size = 0;
public:
    explicit pySurface(const std::string &name) : name(name) {
        this->mesh = new Surface();
        this->mesh->read_obj(name);
        std::cout << 1 << std::endl;
        this->triangles = this->mesh->getTrianglesAsVec();
        std::cout << 2 << std::endl;
        this->points = this->mesh->getPoints();
        std::cout << 3 << std::endl;
        this->neighb_tri = compute_neighbours();
    }
    const std::string &getName() const { return name; }
    void modify_points(std::vector<double> points);
    bool self_intersection_test(const std::vector<double>& new_points);
    void apply_transformation(const std::vector<std::vector<double>>& arr);

    static bool triangles_intersected( std::vector<std::vector<double>> points);
    std::vector<std::set<int>> compute_neighbours();


    virtual ~pySurface() {
        if (mesh!= nullptr) delete mesh;
    }


};


#endif //BAYESSIAN_SEGMENTATION_CPP_PYSURFACE_H
