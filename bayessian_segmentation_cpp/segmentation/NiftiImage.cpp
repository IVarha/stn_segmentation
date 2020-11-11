//
// Created by ivarh on 11/11/2020.
//

#include "NiftiImage.h"

void NiftiImage::read_nifti_image(string file) {
    int gzz = nifti_compiled_with_zlib();
    niimg = nifti_image_read(file.c_str(),1);

}

NiftiImage::~NiftiImage() {
    delete niimg;

}
