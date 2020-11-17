//
// Created by ivarh on 09/11/2020.
//

#ifndef BAYESSIAN_SEGMENTATION_CPP_SURFACE_H
#define BAYESSIAN_SEGMENTATION_CPP_SURFACE_H

#include <string>
#include <list>
#include <vtkPoints.h>
#include <vtkCellArray.h>
#include <vtkPolyData.h>
#include "vtkSmartPointer.h"

class Surface {

    private:

    vtkSmartPointer<vtkPoints> points;
    vtkSmartPointer<vtkCellArray> triangles;
    vtkSmartPointer<vtkPolyData> mesh;
    public:
        void read_volume(const std::string& file_name );
        void expand_volume(double mm);
        void write_volume(const std::string file_name);
        void write_obj(const std::string file_name);
};


#endif //BAYESSIAN_SEGMENTATION_CPP_SURFACE_H
