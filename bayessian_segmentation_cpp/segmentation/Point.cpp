//
// Created by ivarh on 19/11/2020.
//

#include "Point.h"
#include <NiftiImage.h>
Point::Point(const double *pt){
    this->pt[0] = pt[0];
    this->pt[1] = pt[1];
    this->pt[2] = pt[2];
}

//simple realisation of newton step
Point Point::move_point(VolumeDouble &image, Point &direction, double reach_value,double step, double eps) {
    double curr_pt[3];
    curr_pt[0] = this->pt[0];
    curr_pt[1] = this->pt[1];
    curr_pt[2] = this->pt[2];
    double cur_dir = 1;
    double curr_step = step;
    cout << image.getVolume().max();
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
    return this->pt[0];
}

double Point::getY() {
    return this->pt[1];
}

double Point::getZ() {
    return this->pt[2];
}

const double *Point::getPt() const {
    return pt;
}

Point::Point(double x, double y,double z) {
    this->pt[0]= x;
    this->pt[1]= y;
    this->pt[2]= z;
}

Point Point::move_point_with_stop(VolumeDouble &image, Point &direction, Point &stop, double reach_value, double step,
                                  double eps) {
    double curr_pt[3];
    curr_pt[0] = this->pt[0];
    curr_pt[1] = this->pt[1];
    curr_pt[2] = this->pt[2];
    double cur_dir = 1;
    double curr_step = step;
    cout << image.getVolume().max();
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
    double* res = new double(3);
    res[0] = p1[1]*p2[2] - p1[2]*p2[1];
    res[1] = p1[2]*p2[0] - p1[0]*p2[2];
    res[2] = p1[0]*p2[1] - p1[1]*p2[0];
    return res;
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
