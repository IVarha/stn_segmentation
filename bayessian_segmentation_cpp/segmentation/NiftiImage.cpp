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
#include <vtkNIFTIImageReader.h>
#include <vtkSmartPointer.h>
#include <vtkMatrix4x4.h>
#include <vtkImageData.h>
#include <vtkPointData.h>
#include <vtkDataArray.h>
#include <vtkNIFTIImageHeader.h>

using namespace arma;
void NiftiImage::read_nifti_image(string file) {
    int gzz = nifti_compiled_with_zlib();
    auto reader = vtkSmartPointer<vtkNIFTIImageReader>::New();
    auto val = reader->CanReadFile(file.c_str());
    reader->SetFileName(file.c_str());
    reader->Update();
    auto qf = reader->GetQFormMatrix();
    vtkMatrix4x4* sf = reader->GetSFormMatrix();

//    cout << sf->GetElement(0,0) << sf->GetElement(0,3) << endl;
    //niimg = nifti_image_read(file.c_str(),1);

    this->transform = Mat<double>(4,4,fill::eye);
    if (qf != nullptr){

            this->transform(0, 0) = qf->GetElement(0,0);
            this->transform(0, 1) = qf->GetElement(0,1);
            this->transform(0, 2) = qf->GetElement(0,2);
            this->transform(0, 3) = qf->GetElement(0,3);
            this->transform(1, 0) = qf->GetElement(1,0);
            this->transform(1, 1) = qf->GetElement(1,1);
            this->transform(1, 2) = qf->GetElement(1,2);
            this->transform(1, 3) = qf->GetElement(1,3);
            this->transform(2, 0) = qf->GetElement(2,0);
            this->transform(2, 1) = qf->GetElement(2,1);
            this->transform(2, 2) = qf->GetElement(2,2);
            this->transform(2, 3) = qf->GetElement(2,3);
            this->transform(3, 0) = qf->GetElement(3,0);
            this->transform(3, 1) = qf->GetElement(3,1);
            this->transform(3, 2) = qf->GetElement(3,2);
            this->transform(3, 3) = qf->GetElement(3,3);

    }else{
        if (sf != nullptr){
            this->transform(0, 0) = sf->GetElement(0,0);
            this->transform(0, 1) = sf->GetElement(0,1);
            this->transform(0, 2) = sf->GetElement(0,2);
            this->transform(0, 3) = sf->GetElement(0,3);
            this->transform(1, 0) = sf->GetElement(1,0);
            this->transform(1, 1) = sf->GetElement(1,1);
            this->transform(1, 2) = sf->GetElement(1,2);
            this->transform(1, 3) = sf->GetElement(1,3);
            this->transform(2, 0) = sf->GetElement(2,0);
            this->transform(2, 1) = sf->GetElement(2,1);
            this->transform(2, 2) = sf->GetElement(2,2);
            this->transform(2, 3) = sf->GetElement(2,3);
            this->transform(3, 0) = sf->GetElement(3,0);
            this->transform(3, 1) = sf->GetElement(3,1);
            this->transform(3, 2) = sf->GetElement(3,2);
            this->transform(3, 3) = sf->GetElement(3,3);
        }
    }

    //print info about image
    double range[2];
    reader->GetOutput()->GetPointData()->GetScalars()->GetRange(range);

    std::cout << range[0] << ", " << range[1] << std::endl;
    std::cout << reader->GetDataSpacing()[0] << ", " << reader->GetDataSpacing()[1] << " ," << reader->GetDataSpacing()[2] <<  std::endl;
    std::cout << reader->GetDataSpacing()[0] << ", " << reader->GetDataSpacing()[1] << " ," << reader->GetDataSpacing()[2] <<  std::endl;
    auto data = reader->GetOutput();

    auto resp  = data->GetPoint(0);
    cout << data->GetPoint(0)[0] << endl;

    this->niimg = data;
    this->type = data->GetDataObjectType();

    this->xdim = reader->GetDataSpacing()[0];
    this->ydim =reader->GetDataSpacing()[1];
    this->zdim = reader->GetDataSpacing()[2];
    this->type= reader->GetNIFTIHeader()->GetDataType();
    this->nx = data->GetDimensions()[0];
    this->ny = data->GetDimensions()[1];
    this->nz = data->GetDimensions()[2];


//
//    this->type = niimg->datatype;


}

NiftiImage::~NiftiImage() {
    this->niimg->Delete();
}


void* NiftiImage::returnImage() {
    auto trans =  TransformMatrix();
    trans.setMatrix(this->transform);
    this->niimg->GetPointData()->Print(cout);

    switch (this->type){
        case NIFTI_TYPE_INT64:{

            auto* res = new Cube<int>(this->nx,this->ny,this->nz,fill::zeros);
            int ca = 0;
            for (int i = 0;i<this->nx;i++){
                for (int j = 0;j<this->ny;j++){
                    for (int k = 0;k<this->nz;k++){
                        int64_t * pix = static_cast<int64_t*>(this->niimg->GetScalarPointer(i,j,k));

                        res->operator()(i,j,k)= (int)*pix;
                        if (*pix == 6 ) {
                            cout << i << " " << j << " " << k << endl;
                        }
                    }
                }
            }
            VolumeInt* volumeInt = new VolumeInt();
            volumeInt->setVolume(*res);
            volumeInt->setTransformation(trans);
            return volumeInt;
        }
        case NIFTI_TYPE_INT32:{
            auto* res = new Cube<int>(this->nx,this->ny,this->nz,fill::zeros);
            int ca = 0;
            for (int i = 0;i<this->nx;i++){
                for (int j = 0;j<this->ny;j++){
                    for (int k = 0;k<this->nz;k++){
                        int32_t* pix = static_cast<int32_t*>(this->niimg->GetScalarPointer(i,j,k));
                        res->operator()(i,j,k)= (int)*pix;
                        if (*pix == 6) {
                            cout << i << " " << j << " " << k << endl;
                        }
                    }
                }
            }
            VolumeInt* volumeInt = new VolumeInt();
            volumeInt->setVolume(*res);
            volumeInt->setTransformation(trans);
            return volumeInt;
        }
        case NIFTI_TYPE_UINT16:{
            auto* res = new Cube<int>(this->nx,this->ny,this->nz,fill::zeros);
            int ca = 0;
            for (int i = 0;i<this->nx;i++){
                for (int j = 0;j<this->ny;j++){
                    for (int k = 0;k<this->nz;k++){
                        uint16_t * pix = static_cast<uint16_t*>(this->niimg->GetScalarPointer(i,j,k));
                        res->operator()(i,j,k)= (int)*pix;
                        auto cel = this->niimg->GetCell(i,j,k);
                        if ((*pix == 6) and (cel)) {
                            ca++;
                            cout << i << " " << j << " " << k << endl;
                        }
                    }
                }
            }
            VolumeInt* volumeInt = new VolumeInt();
            volumeInt->setVolume(*res);
            volumeInt->setTransformation(trans);
            return volumeInt;
        }
//        case NIFTI_TYPE_FLOAT32:
//        {
//            auto* nii_columns_data = static_cast<float*>(this->niimg->data);
//            auto* res = new Cube<double>(this->nx,this->niimg->ny,this->niimg->nz,fill::zeros);
//
//            for (int cnt = 0; cnt<this->niimg->nvox;cnt++){
//                double val;
//                val = static_cast<double>(*(nii_columns_data + cnt));
//                tie(i,j,k) = ind2sub_3D(cnt, this->niimg->nx,this->niimg->ny);
//                res->operator()(i,j,k)= static_cast<double>(val);
//            }
//            return res;
//        }
//        case NIFTI_TYPE_FLOAT64:
//        {
//            auto* nii_columns_data = static_cast<double*>(this->niimg->data);
//            auto* res = new Cube<double>(this->niimg->nx,this->niimg->ny,this->niimg->nz,fill::zeros);
//
//            for (int cnt = 0; cnt<this->niimg->nvox;cnt++){
//                double val;
//                val = static_cast<double>(*(nii_columns_data + cnt));
//                tie(i,j,k) = ind2sub_3D(cnt, this->niimg->nx,this->niimg->ny);
//                res->operator()(i,j,k)= static_cast<double>(val);
//            }
//            return res;
//        }

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
    voxToScaledWorld(0,0) = this->xdim;
    voxToScaledWorld(1,1) = this->ydim;
    voxToScaledWorld(2,2) = this->zdim;
    voxToScaledWorld(3,3) = 1;

    bool isneuro = (det(this->transform) > 0);

    if (isneuro){
        double x = (this->nx - 1)* this->xdim;
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

TransformMatrix NiftiImage::get_fsl_to_voxel() {
    auto res = this->get_voxel_to_fsl();

    auto result = TransformMatrix();
    result.setMatrix(res.getInverseMat());
    return result;
}

TransformMatrix NiftiImage::get_voxel_to_world() {

    auto res =  TransformMatrix();
    res.setMatrix(this->transform);
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
    return tuple<double, double, double>(ress(0,0),ress(1,0),ress(2,0));
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
    //source_WtoFsl.getMatrix().print("premat");
    //post_Fsl2W.getMatrix().print(" postmat");
    res.setMatrix(post_Fsl2W.matrix*(fslMat.getMatrix() * source_WtoFsl.getMatrix()));
    return res;
}



TransformMatrix TransformMatrix::convert_flirt_W_V(TransformMatrix fslMat, NiftiImage source, NiftiImage reference) {

    TransformMatrix res = TransformMatrix();
    auto source_WtoFsl = source.get_world_to_fsl();
    auto post_Fsl2W = reference.get_fsl_to_voxel();
    //source_WtoFsl.getMatrix().print("premat");
    //post_Fsl2W.getMatrix().print(" postmat");
    res.setMatrix(post_Fsl2W.matrix*(fslMat.getMatrix() * source_WtoFsl.getMatrix()));
    return res;
}

TransformMatrix TransformMatrix::copy() {

    auto res = TransformMatrix();
    res.setMatrix(this->matrix);
    return res;
}

double *TransformMatrix::apply_transform(double x, double y, double z) {

    Mat<double> vect = Mat<double>(4,1);
    vect(0,0) = x;
    vect(1,0) = y;
    vect(2,0) = z;
    vect(3,0) = 1;
    Mat<double> ress = this->matrix * vect;
    double* res = new double(3);
    res[0] = ress(0,0);
    res[1] = ress(1,0);
    res[2] = ress(2,0);
    return res;
}

double *TransformMatrix::apply_transform(const double *pt) {
    Mat<double> vect = Mat<double>(4,1);
    vect(0,0) = pt[0];
    vect(1,0) = pt[1];
    vect(2,0) = pt[2];
    vect(3,0) = 1;
    Mat<double> ress = this->matrix * vect;
    double* res = new double(3);
    res[0] = ress(0,0);
    res[1] = ress(1,0);
    res[2] = ress(2,0);
    return res;
}


double VolumeDouble::interpolate_value_vox(double x, double y, double z, const string& method) {
    if (method == "linear"){//linear
        if (not this->has_slab_mask){
            double x1 = floor(x);
            double y1 = floor(y);
            double z1 = floor(z);
            double xd = x - x1;
            double yd = y - y1;
            double zd = z - z1;
            if (((x + 1) >  this->v.n_rows) or ((y+1)> this->v.n_cols) or ((z+1)> this->v.n_slices) or (x < 0) or (y<0) or (z<0) ) return 0;
            double c000 = this->v((int)(x1),(int)(y1),(int)(z1));
            double c001 = this->v(int(x1),int(y1),int(z1 + 1));
            double c010 = this->v(int(x1),int(y1+1),int(z1));
            double c011 = this->v(int(x1),int(y1+1),int(z1+1));
            double c100 = this->v(int(x1+1),int(y1),int(z1));
            double c101 = this->v(int(x1+1),int(y1),int(z1+1));
            double c110 = this->v(int(x1+1),int(y1+1),int(z1));
            double c111 = this->v(int(x1+1),int(y1+1),int(z1+1));

            double c00 = c000*(1 - xd) + c100*xd;
            double c01 = c001*(1 - xd) + c101*xd;
            double c10 = c010*(1 - xd) + c110*xd;
            double c11 = c011*(1 - xd) + c111*xd;

            double c0 = c00*(1-yd) + c10*yd;
            double c1 = c01*(1-yd) + c11*yd;

            return c0* (1 - zd) + c1*zd;
        }

    }

    if (method == "bicubic"){
        return 0.0;
    }
    return 0.0;
}

const Cube<double> &VolumeDouble::getVolume() const {
    return v;
}

void VolumeDouble::setVolume(const Cube<double> &v) {
    VolumeDouble::v = v;
}

const TransformMatrix &VolumeDouble::getTransformation() const {
    return transformation;
}

void VolumeDouble::setTransformation(const TransformMatrix &transformation) {
    VolumeDouble::transformation = transformation;
}
//VOLUMEINT

VolumeInt VolumeInt::copy(){
    auto res = VolumeInt();
    res.setVolume(this->v);
    res.setTransformation(this->transformation.copy());
    return res;
}

const Cube<int> &VolumeInt::getVolume() const {
    return v;
}

void VolumeInt::setVolume(const Cube<int> &v) {
    VolumeInt::v = v;
}

const TransformMatrix &VolumeInt::getTransformation() const {
    return transformation;
}

void VolumeInt::setTransformation(const TransformMatrix &transformation) {
    VolumeInt::transformation = transformation;
}

VolumeInt::~VolumeInt() {
    this->v.clear();
}

VolumeInt VolumeInt::label_to_mask(int value) {
    auto res = VolumeInt();

    Cube<int> vol = Cube<int>(this->v.n_rows, this->v.n_cols,this->v.n_slices, fill::zeros);
    int cnt = 0;
    for (int i = 0; i< this->v.n_rows;i++){
        for (int j = 0; j< this->v.n_cols;j++){
            for (int k = 0; k< this->v.n_slices;k++){
                if (this->v(i,j,k)== value){
                    vol(i,j,k)=1;
                    cnt = cnt + 1;
                }
            }
        }
    }

    res.setVolume(vol);
    res.setTransformation(this->transformation.copy());
    return res;
}

VolumeDouble VolumeInt::int_to_double() {
    auto res =  VolumeDouble();
    auto cube = Cube<double>(this->v.n_rows,this->v.n_cols,this->v.n_slices,fill::zeros);
    for (int i = 0; i< this->v.n_rows;i++){
        for (int j = 0; j< this->v.n_cols;j++){
            for (int k = 0; k< this->v.n_slices;k++){
                cube(i,j,k)=(double)this->v(i,j,k);

            }
        }
    }
    res.setVolume(cube);
    res.setTransformation(this->transformation.copy());
    return res;
}

Point VolumeInt::center_of_mass() {
    double x_val = 0;double y_val = 0;double z_val = 0;double cnt = 0;
    for (int i = 0; i < this->v.n_rows;i++){
        for (int j = 0; j < this->v.n_cols;j++){
            for (int k = 0; k < this->v.n_slices;k++){
                x_val +=  i*(this->v(i,j,k));
                y_val +=  j*(this->v(i,j,k));
                z_val +=  k*(this->v(i,j,k));
                if (this->v(i,j,k) == 1) cnt += 1;
            }

        }

    }
    auto norm =  arma::accu(this->v);
    x_val = x_val / cnt;
    y_val = y_val / cnt;
    z_val = z_val / cnt;
//    retu

    return Point(x_val,y_val,z_val);
}
