//
// Created by ivarh on 11/11/2020.
//

#ifndef BAYESSIAN_SEGMENTATION_CPP_NIFTIIMAGE_H
#define BAYESSIAN_SEGMENTATION_CPP_NIFTIIMAGE_H

#define NIFTI_TYPE_UINT8           2
/*! signed short. */
#define NIFTI_TYPE_INT16           4
/*! signed int. */
#define NIFTI_TYPE_INT32           8
/*! 32 bit float. */
#define NIFTI_TYPE_FLOAT32        16
/*! 64 bit complex = 2 32 bit floats. */
#define NIFTI_TYPE_COMPLEX64      32
/*! 64 bit float = double. */
#define NIFTI_TYPE_FLOAT64        64
/*! 3 8 bit bytes. */
#define NIFTI_TYPE_RGB24         128
/*! signed char. */
#define NIFTI_TYPE_INT8          256
/*! unsigned short. */
#define NIFTI_TYPE_UINT16        512
/*! unsigned int. */
#define NIFTI_TYPE_UINT32        768
/*! signed long long. */
#define NIFTI_TYPE_INT64        1024
/*! unsigned long long. */
#define NIFTI_TYPE_UINT64       1280
/*! 128 bit float = long double. */
#define NIFTI_TYPE_FLOAT128     1536
/*! 128 bit complex = 2 64 bit floats. */
#define NIFTI_TYPE_COMPLEX128   1792
/*! 256 bit complex = 2 128 bit floats */


#include <string>
#include "nifti2_io.h"

using namespace std;
class TransformMatrix{
private:
    float _data[4][4];
protected:
public:

};

class NiftiImage {
private:
    nifti_image* niimg = nullptr;
protected:
public:

    virtual ~NiftiImage();

    void read_nifti_image(string fileName);
};


#endif //BAYESSIAN_SEGMENTATION_CPP_NIFTIIMAGE_H
