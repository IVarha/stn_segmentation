//
// Created by ivarh on 06/02/2021.
//

#include "pyNiftiImage.h"

#include <utility>

pyNiftiImage::pyNiftiImage(std::string file_name) {


    this->image = new NiftiImage();

    this->image->read_nifti_image(std::move(file_name));


}

void pyNiftiImage::setMask(std::string file_name) {

    this->mask = new NiftiImage();
    this->mask->read_nifti_image(std::move(file_name));

}

pyNiftiImage::~pyNiftiImage() {

    delete this->image;
    delete this->mask;

}

std::vector<std::vector<double>> pyNiftiImage::interpolate_normals(std::vector<std::vector<std::vector<double>>> normals) {
    std::vector<std::vector<double>> res = std::vector<std::vector<double>>();
    int vsize = normals[0].size();
    for (auto & normal : normals){

        auto tmp = std::vector<double>();
        for (int j = 0; j< vsize;j++){
            tmp.push_back(this->image->bSplineInterp(normal[j]));

        }
        res.push_back(tmp);

    }

    return res;
}


