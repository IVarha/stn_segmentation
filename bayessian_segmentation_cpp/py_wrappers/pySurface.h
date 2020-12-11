//
// Created by ivarh on 11/12/2020.
//

#ifndef BAYESSIAN_SEGMENTATION_CPP_PYSURFACE_H
#define BAYESSIAN_SEGMENTATION_CPP_PYSURFACE_H
#include <string>
#include "Surface.h"
class pySurface {
    std::string name;

    Surface* mesh = nullptr;
    int p_size = 0;
public:
    explicit pySurface(const std::string &name) : name(name) {
        this->mesh = new Surface();
        this->mesh->read_obj(name);
    }
    const std::string &getName() const { return name; }
    void modify_points(std::vector<double> points);
    bool self_intersection_test(std::vector<double> points);


    virtual ~pySurface() {
        if (mesh!= nullptr) delete mesh;
    }


};


#endif //BAYESSIAN_SEGMENTATION_CPP_PYSURFACE_H
