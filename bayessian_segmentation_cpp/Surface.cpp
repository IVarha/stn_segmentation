//
// Created by ivarh on 09/11/2020.
//

#include "Surface.h"
#include "ostream"
#include <vtkPoints.h>
#include <vtkPolyDataReader.h>
#include <vtkSmartPointer.h>
#include <vtkPolyDataMapper.h>
#include <vtkStructuredPointsReader.h>
#include <vtkImageDataGeometryFilter.h>
#include <vtkRenderWindow.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkRenderer.h>
#include <sstream>


void Surface::read_volume(const std::string& file_positions) {
    int s = 1;
    auto reader = vtkSmartPointer<vtkPolyDataReader>::New();
    reader->SetFileName(file_positions.c_str());
    reader->Update();

    if (int errorcode = reader->GetErrorCode())
    {
        std::ostringstream what;
        what << "vtkPolyDataReader() failed with error code " << errorcode << ".";
    }

    // We just want the points and polys, no scalars/normals/etc that might be in the file
    vtkSmartPointer<vtkPolyData> origpd = reader->GetOutput();
    auto points = vtkSmartPointer<vtkPoints>::New();
    auto polys = vtkSmartPointer<vtkCellArray>::New();
    auto newpd = vtkSmartPointer<vtkPolyData>::New();
    points->DeepCopy(origpd->GetPoints());
    polys->DeepCopy(origpd->GetPolys());
    newpd->SetPoints(points);
    newpd->SetPolys(polys);


}
