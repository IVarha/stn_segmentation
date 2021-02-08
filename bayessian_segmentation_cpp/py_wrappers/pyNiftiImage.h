//
// Created by ivarh on 06/02/2021.
//

#ifndef BAYESSIAN_SEGMENTATION_CPP_PYNIFTIIMAGE_H
#define BAYESSIAN_SEGMENTATION_CPP_PYNIFTIIMAGE_H


#include <NiftiImage.h>

class pyNiftiImage {

    NiftiImage* image;
    NiftiImage* mask;


public:
    explicit pyNiftiImage(std::string file_name);

    void setMask(std::string file_name);

    std::vector<std::vector<double>> interpolate_normals(std::vector<std::vector<std::vector<double>>> normals);

    virtual ~pyNiftiImage();
};


#endif //BAYESSIAN_SEGMENTATION_CPP_PYNIFTIIMAGE_H
