//
// Created by ivarh on 19/11/2020.
//

#ifndef BAYESSIAN_SEGMENTATION_CPP_POINT_H
#define BAYESSIAN_SEGMENTATION_CPP_POINT_H


#include <tuple>

class VolumeDouble;

class Point {

public:
    const double *getPt() const;

private:
    double pt[3];
    public:
    explicit Point(const double *pt);
    Point(double x,double y,double z);
    double getX();
    double getY();
    double getZ();
    Point move_point(VolumeDouble &image, Point &direction, double reach_value,double step ,double eps);
    Point move_point_with_stop(VolumeDouble &image, Point &direction, Point &stop,double reach_value,double step ,double eps);
    std::tuple<double,double,double> to_tuple();
    static double* cross_product(double* p1, double* p2);
    static double* substract(const double* V1,const double* V2);
    static double scalar(const double* V1,const double* V2);
    static bool isEqual(const double* V1,const double* V2);
};


#endif //BAYESSIAN_SEGMENTATION_CPP_POINT_H
