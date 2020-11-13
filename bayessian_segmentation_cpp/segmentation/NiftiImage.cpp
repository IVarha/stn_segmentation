//
// Created by ivarh on 11/11/2020.
//
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

#include "NiftiImage.h"
#include <armadillo>
#include "laynii_lib.h"
#include <tuple>
using namespace arma;
void NiftiImage::read_nifti_image(string file) {
    int gzz = nifti_compiled_with_zlib();
    niimg = nifti_image_read(file.c_str(),1);

    this->transform = Mat<double>(4,4,fill::eye);

    this->transform(0,0) = niimg->sto_xyz.m[0][0];
    this->transform(0,1) = niimg->sto_xyz.m[0][1];
    this->transform(0,2) = niimg->sto_xyz.m[0][2];
    this->transform(0,3) = niimg->sto_xyz.m[0][3];
    this->transform(1,0) = niimg->sto_xyz.m[1][0];
    this->transform(1,1) = niimg->sto_xyz.m[1][1];
    this->transform(1,2) = niimg->sto_xyz.m[1][2];
    this->transform(1,3) = niimg->sto_xyz.m[1][3];
    this->transform(2,0) = niimg->sto_xyz.m[2][0];
    this->transform(2,1) = niimg->sto_xyz.m[2][1];
    this->transform(2,2) = niimg->sto_xyz.m[2][2];
    this->transform(2,3) = niimg->sto_xyz.m[2][3];
    this->transform(3,0) = niimg->sto_xyz.m[3][0];
    this->transform(3,1) = niimg->sto_xyz.m[3][1];
    this->transform(3,2) = niimg->sto_xyz.m[3][2];
    this->transform(3,3) = niimg->sto_xyz.m[3][3];

    this->xdim = niimg->dx;
    this->ydim = niimg->dy;
    this->zdim = niimg->dz;

    this->type = niimg->datatype;


}

NiftiImage::~NiftiImage() {
    delete niimg;

}


void* NiftiImage::returnImage() {
    int i,j,k;

    switch (this->type){
        case NIFTI_TYPE_INT64:{
            auto* nii_columns_data = static_cast<int64_t*>(this->niimg->data);
            auto* res = new Cube<int>(this->niimg->nx,this->niimg->ny,this->niimg->nz,fill::zeros);

            for (int cnt = 0; cnt<this->niimg->nvox;cnt++){
                int16_t val;
                val = static_cast<int16_t >(*(nii_columns_data + cnt));
                tie(i,j,k) = ind2sub_3D(cnt, this->niimg->nx,this->niimg->ny);
                res->operator()(i,j,k)= static_cast<int>(val);
                if (val == 6){
                    cout<< i << " " << j << " " << k << endl;
                }
                cnt++;
            }
            return res;
        }
        case NIFTI_TYPE_INT32:{
            auto* nii_columns_data = static_cast<int32_t*>(this->niimg->data);
            auto* res = new Cube<int>(this->niimg->nx,this->niimg->ny,this->niimg->nz,fill::zeros);

            for (int cnt = 0; cnt<this->niimg->nvox;cnt++){
                int16_t val;
                val = static_cast<int16_t >(*(nii_columns_data + cnt));
                tie(i,j,k) = ind2sub_3D(cnt, this->niimg->nx,this->niimg->ny);
                res->operator()(i,j,k)= static_cast<int>(val);
                cnt++;
            }
            return res;
        }
        case NIFTI_TYPE_FLOAT32:
        {
            auto* nii_columns_data = static_cast<float*>(this->niimg->data);
            auto* res = new Cube<double>(this->niimg->nx,this->niimg->ny,this->niimg->nz,fill::zeros);

            for (int cnt = 0; cnt<this->niimg->nvox;cnt++){
                double val;
                val = static_cast<double>(*(nii_columns_data + cnt));
                tie(i,j,k) = ind2sub_3D(cnt, this->niimg->nx,this->niimg->ny);
                res->operator()(i,j,k)= static_cast<double>(val);
                cnt++;
            }
            return res;
        }
        case NIFTI_TYPE_FLOAT64:
        {
            auto* nii_columns_data = static_cast<double*>(this->niimg->data);
            auto* res = new Cube<double>(this->niimg->nx,this->niimg->ny,this->niimg->nz,fill::zeros);

            for (int cnt = 0; cnt<this->niimg->nvox;cnt++){
                double val;
                val = static_cast<double>(*(nii_columns_data + cnt));
                tie(i,j,k) = ind2sub_3D(cnt, this->niimg->nx,this->niimg->ny);
                res->operator()(i,j,k)= static_cast<double>(val);
                cnt++;
            }
            return res;
        }

    }

    return new Mat<float>(1,1);
}

const Mat<float> &TransformMatrix::getMatrix() const {
    return matrix;
}

void TransformMatrix::setMatrix(const Mat<float> &matrix) {
    TransformMatrix::matrix = matrix;
    this->inverse_mat = inv(matrix);
}

const Mat<float> &TransformMatrix::getInverseMat() const {
    return inverse_mat;
}

tuple<float, float, float> TransformMatrix::mm_to_vox(float x, float y, float z) {
    Mat<float> vect = Mat<float>(4,1);
    vect(0,0) = x;
    vect(1,0) = y;
    vect(2,0) = z;
    vect(3,0) = 1;
    Mat<float> ress = this->inverse_mat * vect;
    return tuple<float, float, float>(ress(0,0),ress(0,1),ress(0,2));
}

tuple<float, float, float> TransformMatrix::vox_to_mm(int x, int y, int z) {
    Mat<float> vect = Mat<float>(4,1);
    vect(0,0) = x;
    vect(1,0) = y;
    vect(2,0) = z;
    vect(3,0) = 1;
    Mat<float> ress = this->matrix * vect;
    return tuple<float, float, float>(ress(0,0),ress(0,1),ress(0,2));
}

double VolumeDouble::interpolate_value_mm(float x, float y, float z, string method) {
    return 0;
}
