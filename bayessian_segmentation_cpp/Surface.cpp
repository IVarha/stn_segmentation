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
#include <vtkPolyDataNormals.h>
#include <vtkPolyData.h>
#include <vtkPointData.h>
#include <vtkPolyDataWriter.h>
#include <iostream>
#include <vtkSTLWriter.h>
#include <vtkOBJWriter.h>
#include <vtkSphere.h>
#include <vtkSphereSource.h>
#include <vtkLinearSubdivisionFilter.h>
#include <vtkCenterOfMass.h>
#include <vtkSmoothPolyDataFilter.h>
#include <Point.h>
#include <vtkMassProperties.h>


void Surface::read_volume(const std::string& file_name ) {
    int s = 1;
    auto reader = vtkSmartPointer<vtkPolyDataReader>::New();
    reader->SetFileName(file_name.c_str());
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
//    double x = points->GetPoint(1)[0];
//    double y = points->GetPoint(1)[1];
//    double z = points->GetPoint(1)[2];

    this->points = points;
    this->triangles = polys;
    //auto* list = vtkIdList::New();
    //this->triangles->GetCellAtId(1,list);
    //list->Delete();
    //int a = list->GetId(0);
    //a = list->GetId(1);
    //a = list->GetId(2);
    this->mesh = newpd;
   // cout << points->GetPoint(1)[0] << points->GetPoint(1)[1] << points->GetPoint(1)[2];
}

void Surface::expand_volume(double mm) {
    auto normalsGen =  vtkSmartPointer<vtkPolyDataNormals>::New();
    normalsGen->SetInputData(this->mesh);
    normalsGen->ComputeCellNormalsOff();
    normalsGen->ComputePointNormalsOn();
    bool a1 = normalsGen->GetAutoOrientNormals();
    normalsGen->SetAutoOrientNormals(true);
    normalsGen->Update();
    auto normals = normalsGen->GetOutput();
    auto normals4= normals->GetPointData()->GetNormals();
//    double x = normals4->GetTuple(0)[0];
//    double y = normals4->GetTuple(0)[1];
//    double z = normals4->GetTuple(0)[2];
    normals->GetPointData();
    cout<< "prtdsffsf";
    //new points
    auto new_pts = vtkSmartPointer<vtkPoints>::New();
    for (int i = 0;i<this->points->GetNumberOfPoints();i++){

        double* vox = new double[3];
        double* tmp_pt = this->points->GetPoint(i);
        double* tmp_norm = normals4->GetTuple(i);
        vox[0] = 0 + tmp_pt[0] + mm* tmp_norm[0];
        vox[1] = 0 + tmp_pt[1] + mm* tmp_norm[1];
        vox[2] = 0 + tmp_pt[2] + mm* tmp_norm[2];
        new_pts->InsertNextPoint(vox);
    }

    this->mesh->Initialize();
    this->mesh->SetPolys(this->triangles);
    this->points->Initialize();
    this->points->DeepCopy(new_pts);
    this->mesh->SetPoints(this->points);


}

void Surface::write_volume(const std::string file_name) {
    auto writer = vtkSmartPointer<vtkPolyDataWriter>::New();
    writer->SetFileName(file_name.c_str());
    writer->DebugOn();
    writer->SetInputData(this->mesh);
    writer->Write();
}

void Surface::write_obj(const std::string file_name) {
    auto writer = vtkSmartPointer<vtkOBJWriter>::New();
    writer->SetFileName(file_name.c_str());
    writer->SetInputData(this->mesh);
    writer->Write();
}

void Surface::write_stl(const std::string file_name) {
    auto writer = vtkSmartPointer<vtkSTLWriter>::New();
    writer->SetFileName(file_name.c_str());
    writer->SetInputData(this->mesh);
    writer->Write();
}

Surface Surface::generate_sphere(double radius_mm, std::tuple<double, double, double> center) {

    vtkSmartPointer<vtkSphereSource> sphereSource =
            vtkSmartPointer<vtkSphereSource>::New();
    sphereSource->SetRadius(radius_mm);
    sphereSource->SetCenter(std::get<0>(center),std::get<1>(center),std::get<2>(center));
    sphereSource->Update();
    auto originalMesh = sphereSource->GetOutput();
    std::cout << "Before subdivision" << std::endl;
    std::cout << "    There are " << originalMesh->GetNumberOfPoints()
              << " points." << std::endl;
    std::cout << "    There are " << originalMesh->GetNumberOfPolys()
              << " triangles." << std::endl;
    auto subdivisionFilter = vtkSmartPointer<vtkLinearSubdivisionFilter>::New();
    subdivisionFilter->SetInputData(originalMesh);
    subdivisionFilter->SetNumberOfSubdivisions(2);
    subdivisionFilter->Update();
    auto mesh = subdivisionFilter->GetOutput();
    Surface result = Surface();
    result.setPoints(mesh->GetPoints());
    std::cout << "    There are " << mesh->GetPoints()->GetNumberOfPoints()
              << " points." << std::endl;
    result.setTriangles(mesh->GetPolys());
    result.mesh = mesh;

    return result;
}

const vtkSmartPointer<vtkPoints> &Surface::getPoints() const {
    return points;
}

void Surface::setPoints(const vtkSmartPointer<vtkPoints> &points) {
    Surface::points = points;
}

const vtkSmartPointer<vtkCellArray> &Surface::getTriangles() const {
    return triangles;
}

void Surface::setTriangles(const vtkSmartPointer<vtkCellArray> &triangles) {
    Surface::triangles = triangles;
}

const vtkSmartPointer<vtkPolyData> &Surface::getMesh() const {
    return mesh;
}

void Surface::setMesh(const vtkSmartPointer<vtkPolyData> &mesh) {
    Surface::mesh = mesh;
}

std::tuple<double, double, double> Surface::centre_of_mesh() {

    auto centre = vtkSmartPointer<vtkCenterOfMass>::New();
    centre->SetInputData(this->mesh);
    centre->Update();
    auto res = centre->GetCenter();

    return {res[0],res[1],res[2]};
}

void Surface::shrink_sphere (VolumeDouble &mask, std::tuple<double, double, double> center, double threshold)  {
    auto normalsGen =  vtkSmartPointer<vtkPolyDataNormals>::New();
    normalsGen->SetInputData(this->mesh);
    normalsGen->ComputeCellNormalsOff();
    normalsGen->ComputePointNormalsOn();
    bool a1 = normalsGen->GetAutoOrientNormals();
    normalsGen->SetAutoOrientNormals(true);
    normalsGen->Update();
    auto normals = normalsGen->GetOutput();
    auto normals4= normals->GetPointData()->GetNormals();

    Point center1 = Point(get<0>(center),get<1>(center),get<2>(center));

    double nm;
    for(int i = 0; i < this->points->GetNumberOfPoints(); i++){
        auto vox = Point(this->points->GetPoint(i));
        double* normal = normals4->GetTuple(i);

        normal[0] = get<0>(center) - vox.getX();
        normal[1] = get<1>(center) - vox.getY();
        normal[2] = get<2>(center) - vox.getZ();
        nm = sqrt(normal[0]*normal[0] + normal[1]*normal[1] + normal[2]*normal[2]);
        normal[0] = normal[0]/nm;
        normal[1] = normal[1]/nm;
        normal[2] = normal[2]/nm;
        auto norm2 = Point(normal);
        Point res_vox= vox.move_point_with_stop(mask,norm2,center1,threshold,0.3, 0.01);

        this->points->SetPoint(i,res_vox.getPt());
        cout<<  "Process point :"<< i << endl;
    }

    this->mesh->Initialize();
    this->mesh->SetPoints(this->points);
    this->mesh->SetPolys(this->triangles);
}

void Surface::apply_transformation(TransformMatrix& pre_transformation) {

    for (int i = 0;i< this->points->GetNumberOfPoints();i++){
        double* pt = this->points->GetPoint(i);
        double * vox = pre_transformation.apply_transform(pt);
        this->points->SetPoint(i,vox);

    }

    this->mesh->Initialize();
    this->mesh->SetPoints(this->points);
    this->mesh->SetPolys(this->triangles);
}

void Surface::smoothMesh() {
    vtkSmartPointer<vtkSmoothPolyDataFilter> smoothFilter =
            vtkSmartPointer<vtkSmoothPolyDataFilter>::New();
    smoothFilter->SetInputData(this->mesh);

    smoothFilter->SetNumberOfIterations(30);
    smoothFilter->FeatureEdgeSmoothingOff();
    smoothFilter->BoundarySmoothingOn();
    smoothFilter->SetRelaxationFactor(0.1);
    smoothFilter->Update();
    this->points->Initialize();
    this->points->DeepCopy(smoothFilter->GetOutput()->GetPoints());
    this->mesh->Initialize();
    this->mesh->SetPoints(this->points);
    this->mesh->SetPolys(this->triangles);


}

double Surface::calculate_volume() {
    auto mass_calc = vtkSmartPointer<vtkMassProperties>::New();
    mass_calc->SetInputData(this->mesh);
    mass_calc->Update();
    return mass_calc->GetVolume();

}
