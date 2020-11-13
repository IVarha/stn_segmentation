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
#include <armadillo>
using namespace std;
using namespace arma;


class TransformMatrix{
public:
    const Mat<float> &getMatrix() const;

    void setMatrix(const Mat<float> &matrix);

    const Mat<float> &getInverseMat() const;



private:
    Mat<float> matrix;
    Mat<float> inverse_mat;
protected:
public:
    tuple<float, float, float> vox_to_mm(int x, int y,int z);
    tuple<float, float, float> mm_to_vox(float x,float y,float z);


};
class Volume{
protected:
    TransformMatrix transform;
};




class VolumeInt: Volume{
    Cube<int> v;
};


class VolumeDouble: Volume{

public:
    double interpolate_value_mm(float x,float y, float z, string method);
private:
    Cube<double> v;
    TransformMatrix transformation;
};


class NiftiImage {
private:
    nifti_image* niimg = nullptr;
    int type;
    Mat<double> transform;
    double xdim;
    double ydim;
    double zdim;

protected:
public:

    virtual ~NiftiImage();

    void read_nifti_image(string fileName);
    void* returnImage();
};




#endif //BAYESSIAN_SEGMENTATION_CPP_NIFTIIMAGE_H
