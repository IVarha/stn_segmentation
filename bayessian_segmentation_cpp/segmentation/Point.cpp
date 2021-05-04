//
// Created by ivarh on 19/11/2020.
//

#include "Point.h"
#include <NiftiImage.h>
#include "stdlib.h"



Point::Point(const double *pt){
    this->x = pt[0];
    this->y = pt[1];
    this->z = pt[2];
}

//simple realisation of newton step
Point Point::move_point(VolumeDouble &image, Point &direction, double reach_value,double step, double eps) {
    double curr_pt[3];
    curr_pt[0] = x;
    curr_pt[1] = y;
    curr_pt[2] = z;
    double cur_dir = 1;
    double curr_step = step;
    //cout << image.getVolume().max();
    //eval
    double val_pre = image.interpolate_value_vox(curr_pt[0],curr_pt[1],curr_pt[2],"linear");
    double err = abs(val_pre - reach_value);
    while (err > eps){

        double val1 = image.interpolate_value_vox(curr_pt[0] +  curr_step*cur_dir*direction.getX(),
                                                  curr_pt[1] + curr_step*cur_dir*direction.getY(),
                                                  curr_pt[2] + curr_step*cur_dir*direction.getZ(),"linear");

        curr_pt[0] = curr_pt[0] +  curr_step*cur_dir*direction.getX();
        curr_pt[1] = curr_pt[1] +  curr_step*cur_dir*direction.getY();
        curr_pt[2] = curr_pt[2] +  curr_step*cur_dir*direction.getZ();

        if (((val1> reach_value) and (val_pre < reach_value)) or ((val1< reach_value) and (val_pre > reach_value))){
            curr_step = curr_step/2;
            cur_dir = cur_dir * (-1);

        } else{
            if (val1 == reach_value){
                break;
            }
        }
        err = abs(val1 - reach_value);
        val_pre = val1;

    }

    Point res = Point(curr_pt);

    return res;
}

double Point::getX() {
    return x;
}

double Point::getY() {
    return y;
}

double Point::getZ() {
    return z;
}

const double *Point::getPt() const {
    double* res = new double (3);
    res[0] = this->x;
    res[1] = this->y;
    res[2] = this->z;
    return res;
}

Point::Point(double x, double y,double z) {
    this->x= x;
    this->y= y;
    this->z= z;
}

Point Point::move_point_with_stop(VolumeDouble &image, Point &direction, Point &stop, double reach_value, double step,
                                  double eps) {
    double curr_pt[3];
    curr_pt[0] = this->x;
    curr_pt[1] = this->y;
    curr_pt[2] = this->z;
    double cur_dir = 1;
    double curr_step = step;
   // cout << image.getVolume().max();
    //eval
    double val_pre = image.interpolate_value_vox(curr_pt[0],curr_pt[1],curr_pt[2],"linear");
    double err = abs(val_pre - reach_value);
    while (err > eps){

        double val1 = image.interpolate_value_vox(curr_pt[0] +  curr_step*cur_dir*direction.getX(),
                                                  curr_pt[1] + curr_step*cur_dir*direction.getY(),
                                                  curr_pt[2] + curr_step*cur_dir*direction.getZ(),"linear");

        double t_err = sqrt(pow(curr_pt[0] - stop.getX(),2) + pow(curr_pt[1] - stop.getY(),2)
                                                                               + pow(curr_pt[2] - stop.getZ(),2));
        if (t_err < curr_step) break;
        curr_pt[0] = curr_pt[0] +  curr_step*cur_dir*direction.getX();
        curr_pt[1] = curr_pt[1] +  curr_step*cur_dir*direction.getY();
        curr_pt[2] = curr_pt[2] +  curr_step*cur_dir*direction.getZ();

        if (((val1> reach_value) and (val_pre < reach_value)) or ((val1< reach_value) and (val_pre > reach_value))){
            curr_step = curr_step/2;
            cur_dir = cur_dir * (-1);

        } else{
            if (val1 == reach_value){
                break;
            }
        }
        err = abs(val1 - reach_value);
        val_pre = val1;


    }

    Point res = Point(curr_pt);
    return res;
}

std::tuple<double, double, double> Point::to_tuple() {
    return std::tuple<double, double, double>(this->getX(), this->getY(), this->getZ());
}

double* Point::cross_product(double *p1, double *p2) {
    auto* res = new double(3);
    res[0] = p1[1]*p2[2] - p1[2]*p2[1];
    res[1] = p1[2]*p2[0] - p1[0]*p2[2];
    res[2] = p1[0]*p2[1] - p1[1]*p2[0];
    return res;
}

Point Point::cross_product(Point& p1, Point& p2) {
    return Point(p1.y*p2.z - p1.z*p2.y,
    p1.z*p2.x - p1.x*p2.z,
    p1.x*p2.y - p1.y*p2.x);
}

double *Point::substract(const double *V1,const double *V2) {
    double* res = new double (3);
    res[0] = V1[0] - V2[0];
    res[1] = V1[1] - V2[1];
    res[2] = V1[2] - V2[2];
    return res;
}

double Point::scalar(const double *V1,const double *V2) {
    double res = 0;
    res+= V1[0] * V2[0];
    res+= V1[1] * V2[1];
    res+= V1[2] * V2[2];
    return res;

}

bool Point::isEqual(const double *V1, const double *V2) {
    return ((V1[0]==V2[0])&&(V1[1]==V2[1])&&(V1[2]==V2[2]));
}

double *Point::sum(const double *V1, const double *V2) {
    double* res = new double (3);
    res[0] = V1[0] + V2[0];
    res[1] = V1[1] + V2[1];
    res[2] = V1[2] + V2[2];
    return res;
}

Point Point::operator-(const Point &b) {
    return Point(this->x - b.x, this->y - b.y,this->z - b.z);
}

Point Point::operator+(const Point &b) {
    return Point(this->x + b.x, this->y + b.y,this->z + b.z);
}

Point Point::operator/(double b) {
    return Point(this->x / b, this->y / b,this->z / b);
}

double Point::dot(Point a, Point b) {
    return  a.x*b.x + a.y*b.y + a.z*b.z ;
}

Point Point::operator*(double b) {
    return Point(this->x * b, this->y * b,this->z * b);
}

Point Point::move_in_value_dir(VolumeDouble &image, Point &direction, double step, double  thresh) {
    double curr_pt[3];
    curr_pt[0] = x;
    curr_pt[1] = y;
    curr_pt[2] = z;

    //cout << image.getVolume().max();
    //eval
    double val_pre = image.interpolate_value_vox(curr_pt[0],curr_pt[1],curr_pt[2],"linear");


    double val1 = image.interpolate_value_vox(curr_pt[0] +  step*direction.getX(),
                                                  curr_pt[1] + step*direction.getY(),
                                                  curr_pt[2] + step*direction.getZ(),"linear");

    double val2 = image.interpolate_value_vox(curr_pt[0] -  step*direction.getX(),
                                              curr_pt[1] - step*direction.getY(),
                                              curr_pt[2] - step*direction.getZ(),"linear");

    if ( abs(val1 - thresh) < abs(val_pre - thresh))  {
        return Point( curr_pt[0] +  step*direction.getX(),
                      curr_pt[1] +  step*direction.getY(),
                      curr_pt[2] +  step*direction.getZ());

    } else{
        if (abs(val1 - 1)<0.1) {
            return Point(curr_pt[0] + 5*step * direction.getX(),
                         curr_pt[1] + 5*step * direction.getY(),
                         curr_pt[2] + 5*step * direction.getZ());
        }
        if ( abs(val2 - thresh) < abs(val_pre - thresh))  {
            return Point( curr_pt[0] -  step*direction.getX(),
                          curr_pt[1] -  step*direction.getY(),
                          curr_pt[2] -  step*direction.getZ());}

//        return Point( curr_pt[0] -  step*direction.getX(),
//                      curr_pt[1] -  step*direction.getY(),
//                      curr_pt[2] -  step*direction.getZ());
        return Point( curr_pt[0] ,
                      curr_pt[1] ,
                      curr_pt[2] );

    }
}

Point::Point() {
    this->x = 0;
    this->y = 0;
    this->z = 0;
}

double Point::scalar(const Point& V1, const Point& V2) {
    return V1.x*V2.x + V1.y*V2.y + V1.z*V2.z;
}

Point::Point(std::vector<double> pt) {
    this->x = pt[0];
    this->y = pt[1];
    this->z = pt[2];
}

std::vector<double> Point::toVector() {
    auto res = std::vector<double>();
    res.emplace_back(this->x);
    res.emplace_back(this->y);
    res.emplace_back(this->z);
    return res;
}

double Point::normSquare() {
    return (this->x * this->x + this->y * this->y + this->z * this->z);
}
