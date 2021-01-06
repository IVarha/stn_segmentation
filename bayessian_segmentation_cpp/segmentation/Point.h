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
    static double* cross_product(double* p1, double* p2);
    static double* substract(const double* V1,const double* V2);
    static double scalar(const double* V1,const double* V2);
    static double* sum(const double* V1,const double* V2);
private:
    double x;
    double y;
    double z;




    public:
    explicit Point(const double *pt);
    Point(double x,double y,double z);
    double getX();
    double getY();
    double getZ();
    Point move_point(VolumeDouble &image, Point &direction, double reach_value,double step ,double eps);
    Point move_point_with_stop(VolumeDouble &image, Point &direction, Point &stop,double reach_value,double step ,double eps);
    std::tuple<double,double,double> to_tuple();

    Point operator-(const Point& b);
    Point operator+(const Point& b);
    Point operator/(double b);
    Point operator*(double b);
    static double dot(Point a, Point b);
    static bool isEqual(const double* V1,const double* V2);
};


#endif //BAYESSIAN_SEGMENTATION_CPP_POINT_H
