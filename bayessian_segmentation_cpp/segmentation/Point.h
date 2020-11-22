//
// Created by ivarh on 19/11/2020.
//

#ifndef BAYESSIAN_SEGMENTATION_CPP_POINT_H
#define BAYESSIAN_SEGMENTATION_CPP_POINT_H


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
};


#endif //BAYESSIAN_SEGMENTATION_CPP_POINT_H
