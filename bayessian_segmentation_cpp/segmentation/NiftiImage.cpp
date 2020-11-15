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
#include <iostream>
#include <unordered_map>
using namespace arma;
void NiftiImage::read_nifti_image(string file) {
    int gzz = nifti_compiled_with_zlib();
    niimg = nifti_image_read(file.c_str(),1);

    this->transform = Mat<double>(4,4,fill::eye);
    if ((niimg->qto_xyz.m[0][0]!= 0) and (niimg->qto_xyz.m[0][3]!= 0)) {
        this->transform(0, 0) = niimg->qto_xyz.m[0][0];
        this->transform(0, 1) = niimg->qto_xyz.m[0][1];
        this->transform(0, 2) = niimg->qto_xyz.m[0][2];
        this->transform(0, 3) = niimg->qto_xyz.m[0][3];
        this->transform(1, 0) = niimg->qto_xyz.m[1][0];
        this->transform(1, 1) = niimg->qto_xyz.m[1][1];
        this->transform(1, 2) = niimg->qto_xyz.m[1][2];
        this->transform(1, 3) = niimg->qto_xyz.m[1][3];
        this->transform(2, 0) = niimg->qto_xyz.m[2][0];
        this->transform(2, 1) = niimg->qto_xyz.m[2][1];
        this->transform(2, 2) = niimg->qto_xyz.m[2][2];
        this->transform(2, 3) = niimg->qto_xyz.m[2][3];
        this->transform(3, 0) = niimg->qto_xyz.m[3][0];
        this->transform(3, 1) = niimg->qto_xyz.m[3][1];
        this->transform(3, 2) = niimg->qto_xyz.m[3][2];
        this->transform(3, 3) = niimg->qto_xyz.m[3][3];
    } else{
        if ((niimg->sto_xyz.m[0][0]!= 0) and (niimg->sto_xyz.m[0][3]!= 0)){
            this->transform(0, 0) = niimg->sto_xyz.m[0][0];
            this->transform(0, 1) = niimg->sto_xyz.m[0][1];
            this->transform(0, 2) = niimg->sto_xyz.m[0][2];
            this->transform(0, 3) = niimg->sto_xyz.m[0][3];
            this->transform(1, 0) = niimg->sto_xyz.m[1][0];
            this->transform(1, 1) = niimg->sto_xyz.m[1][1];
            this->transform(1, 2) = niimg->sto_xyz.m[1][2];
            this->transform(1, 3) = niimg->sto_xyz.m[1][3];
            this->transform(2, 0) = niimg->sto_xyz.m[2][0];
            this->transform(2, 1) = niimg->sto_xyz.m[2][1];
            this->transform(2, 2) = niimg->sto_xyz.m[2][2];
            this->transform(2, 3) = niimg->sto_xyz.m[2][3];
            this->transform(3, 0) = niimg->sto_xyz.m[3][0];
            this->transform(3, 1) = niimg->sto_xyz.m[3][1];
            this->transform(3, 2) = niimg->sto_xyz.m[3][2];
            this->transform(3, 3) = niimg->sto_xyz.m[3][3];
        }
    }
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

double NiftiImage::getXdim() const {
    return xdim;
}

double NiftiImage::getYdim() const {
    return ydim;
}

double NiftiImage::getZdim() const {
    return zdim;
}

const Mat<double> &NiftiImage::getTransform() const {
    return transform;
}

TransformMatrix NiftiImage::get_voxel_to_fsl() {
    Mat<double> voxToScaledWorld = Mat<double>(4,4,fill::eye);
    voxToScaledWorld(0,0) = this->niimg->dx;
    voxToScaledWorld(1,1) = this->niimg->dy;
    voxToScaledWorld(2,2) = this->niimg->dz;
    voxToScaledWorld(3,3) = 1;

    bool isneuro = (det(this->transform) > 0);

    if (isneuro){
        double x = (this->niimg->nx - 1)* this->niimg->dx;
        auto tmp = Mat<double>(4,4,fill::eye);
        tmp(0,0) = -1;
        tmp(1,1) = 1;
        tmp(2,2) = 1;
        tmp(0,3) = x;
        tmp(1,3) = 0;
        tmp(2,3) = 0;
        voxToScaledWorld = tmp * voxToScaledWorld;
    }
    TransformMatrix tr = TransformMatrix();
    tr.setMatrix(voxToScaledWorld);
    return tr;
}

TransformMatrix NiftiImage::get_world_to_fsl() {

    auto tran = this->get_voxel_to_fsl();
    auto resmat = tran.getMatrix() *inv(this->transform);

    //Mat<double>* ress = reinterpret_cast<Mat<double>*>((&resmat));

    //potential error
    auto res = TransformMatrix();
    res.setMatrix(resmat);
    return res;
}

TransformMatrix NiftiImage::get_fsl_to_world() {
    auto res =  TransformMatrix();
    auto w2fsl = this->get_world_to_fsl();
    res.setMatrix(w2fsl.getInverseMat());
    return res;
}

const Mat<double> &TransformMatrix::getMatrix() const {
    return matrix;
}

void TransformMatrix::setMatrix(const Mat<double> &matrix) {
    this->matrix = matrix;
    this->inverse_mat = inv(matrix);
}

const Mat<double> &TransformMatrix::getInverseMat() const {
    return inverse_mat;
}

tuple<double, double, double> TransformMatrix::mm_to_vox(double x, double y, double z) {
    Mat<double> vect = Mat<double>(4,1);
    vect(0,0) = x;
    vect(1,0) = y;
    vect(2,0) = z;
    vect(3,0) = 1;
    Mat<double> ress = this->inverse_mat * vect;
    return tuple<double, double, double>(ress(0,0),ress(0,1),ress(0,2));
}

tuple<double, double, double> TransformMatrix::vox_to_mm(int x, int y, int z) {
    Mat<double> vect = Mat<double>(4,1);
    vect(0,0) = x;
    vect(1,0) = y;
    vect(2,0) = z;
    vect(3,0) = 1;
    Mat<double> ress = this->matrix * vect;
    return tuple<double, double, double>(ress(0,0),ress(0,1),ress(0,2));
}

TransformMatrix::~TransformMatrix() {
    this->matrix.clear();
    this->inverse_mat.clear();
}

TransformMatrix TransformMatrix::read_matrix(string fileName) {

    ifstream file;
    file.open(fileName,ios::in);
    string line;

    auto res = TransformMatrix();
    auto mat = Mat<double>(4,4);
    if (file.is_open()){
        int line_num = 0;
        while (getline(file,line)){
            int start = 0;
            int pos = line.find(' ');
            for (int i = 0;i<4;i++){
                mat(line_num,i)= stod(line.substr(start,pos));
                start = pos + 2;
                pos = line.find(' ',start);
            }
            line_num= line_num + 1;
            if (line_num == 4)
                break;
        }
        file.close();
    }
    res.setMatrix(mat);
    return res;
}

TransformMatrix TransformMatrix::convert_flirt_W_W(TransformMatrix fslMat,NiftiImage source,NiftiImage reference)
{
//    //testcode
//    Mat<double> a1 = Mat<double>(2,2,fill::zeros);
//    a1(0,0) = 2;a1(0,1) = 3;a1(1,0) = 6;a1(1,1) = 4;
//    Mat<double> a2 = Mat<double>(2,1,fill::zeros);
//    a2(0,0) = 1;a2(1,0) = 0;
//    Mat<double> a3 = a1*a2;
//    a3.print("a3: ");
//    //endtest

    TransformMatrix res = TransformMatrix();
    auto source_WtoFsl = source.get_world_to_fsl();
    auto post_Fsl2W = reference.get_fsl_to_world();
    source_WtoFsl.getMatrix().print("premat");
    post_Fsl2W.getMatrix().print(" postmat");
    res.setMatrix(post_Fsl2W.matrix*(fslMat.getMatrix() * source_WtoFsl.getMatrix()));
    return res;
}

double VolumeDouble::interpolate_value_mm(double x, double y, double z, string method) {
    return 0.0;
}


