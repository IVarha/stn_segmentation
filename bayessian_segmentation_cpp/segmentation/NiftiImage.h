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


class NiftiImage;

class TransformMatrix{
public:
    const arma::Mat<double> &getMatrix() const;

    void setMatrix(const arma::Mat<double> &matrix);

    const arma::Mat<double> &getInverseMat() const;

    static TransformMatrix read_matrix(string fileName);

    static TransformMatrix convert_flirt_W_W(TransformMatrix fslMat,NiftiImage source,NiftiImage reference );

private:
    arma::Mat<double> matrix;
    arma::Mat<double> inverse_mat;
protected:
public:
    tuple<double, double, double> vox_to_mm(int x, int y,int z);
    tuple<double, double, double> mm_to_vox(double x,double y,double z);

    virtual ~TransformMatrix();


};
class Volume{
protected:
    TransformMatrix transform;
};




class VolumeInt: Volume{
    arma::Cube<int> v;
};


class VolumeDouble: Volume{

public:
    double interpolate_value_mm(double x,double y, double z, string method);
private:
    arma::Cube<double> v;
    TransformMatrix transformation;
};


class NiftiImage {
private:
    nifti_image* niimg = nullptr;
    int type;
    arma::Mat<double> transform;
public:
    const arma::Mat<double> &getTransform() const;

public:
    double getXdim() const;

    double getYdim() const;

    double getZdim() const;

private:
    double xdim;
    double ydim;
    double zdim;

protected:
public:

    virtual ~NiftiImage();

    void read_nifti_image(string fileName);
    void* returnImage();

    TransformMatrix get_voxel_to_fsl();
    TransformMatrix get_world_to_fsl();
    TransformMatrix get_fsl_to_world();
};




#endif //BAYESSIAN_SEGMENTATION_CPP_NIFTIIMAGE_H
