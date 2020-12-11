//
// Created by ivarh on 11/12/2020.
//

#include "pySurface.h"

void pySurface::modify_points(std::vector<double> points) {
    this->mesh->apply_points(points);
}

bool pySurface::self_intersection_test(std::vector<double> points) {
    return false;
}

