//
// Created by ivarh on 11/12/2020.
//

#ifndef BAYESSIAN_SEGMENTATION_CPP_PYSURFACE_H
#define BAYESSIAN_SEGMENTATION_CPP_PYSURFACE_H
#include <string>
#include "Surface.h"
class pySurface {
    std::string name;
public:
    explicit pySurface(const std::string &name) : name(name) { }
    const std::string &getName() const { return name; }






};


#endif //BAYESSIAN_SEGMENTATION_CPP_PYSURFACE_H
