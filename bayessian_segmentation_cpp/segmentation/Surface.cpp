//
// Created by ivarh on 09/11/2020.
//

#include "Surface.h"
#include "ostream"
#include <vtkPoints.h>
#include <vtkPolyDataReader.h>
#include <vtkSmartPointer.h>
#include <vtkPolyDataMapper.h>
#include <cstdio>

#include <vtkRenderWindowInteractor.h>
#include <vtkRenderer.h>
#include <sstream>
#include <vtkPolyDataNormals.h>
#include <vtkPolyData.h>
#include <vtkPointData.h>

#include <vtkWindowToImageFilter.h>
#include <vtkPNGWriter.h>
#include <vtkActor.h>
#include <vtkRenderWindow.h>
#include <vtkOpenGLRenderWindow.h>


#include <vtkPolyDataWriter.h>
#include <iostream>
#include <vtkSTLWriter.h>
#include <vtkOBJWriter.h>
#include <vtkSphere.h>
#include <vtkLinearSubdivisionFilter.h>
#include <vtkCenterOfMass.h>
#include <vtkSmoothPolyDataFilter.h>
#include <Point.h>
#include <vtkMassProperties.h>
#include <vtkSelectEnclosedPoints.h>
#include <vtkOBJReader.h>
#include <vtkSphereSource.h>
#include <vtkCellCenters.h>
#include <vtkImplicitPolyDataDistance.h>
#include "math.h"
#include "spdlog/spdlog.h"
#include "spdlog/cfg/env.h" // support for loading levels from the environment variable
#include "spdlog/sinks/rotating_file_sink.h"
#include <algorithm>

void Surface::read_volume(const std::string& file_name ) {
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
    //cout<< "prtdsffsf";
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
    //writer->DebugOn();
    writer->SetInputData(this->mesh);
    writer->Write();
}

void Surface::write_obj(const std::string file_name) {
    try{
        remove(file_name.c_str());
    }catch (...){

    }

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

Surface Surface::generate_sphere(double radius_mm, std::tuple<double, double, double> center,int subdivisions) {

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
    //subdivisionFilter->SetNumberOfSubdivisions(2);
    subdivisionFilter->SetNumberOfSubdivisions(subdivisions);
    subdivisionFilter->Update();
    auto mesh = subdivisionFilter->GetOutput();
    Surface result = Surface();
    result.points = mesh->GetPoints();
    std::cout << "    There are " << mesh->GetPoints()->GetNumberOfPoints()
              << " points." << std::endl;
    result.triangles = mesh->GetPolys();
    result.mesh = mesh;
    //renew sphere
    result.vec_tri = result.getTrianglesAsVec();
    result.compute_points_neigbours();
    result.compute_tri_neigbours();

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
        //cout<<  "Process point :"<< i << endl;
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

void Surface::smoothMesh(int iter) {
    vtkSmartPointer<vtkSmoothPolyDataFilter> smoothFilter =
            vtkSmartPointer<vtkSmoothPolyDataFilter>::New();
    smoothFilter->SetInputData(this->mesh);

    smoothFilter->SetNumberOfIterations(iter);
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

void Surface::apply_points(std::vector<double>& set_of_points) {

    for (int i = 0;i< this->points->GetNumberOfPoints();i++){
        this->points->SetPoint(i,set_of_points[3*i],set_of_points[3*i +1],set_of_points[3*i +2]);
    }
    this->points->Modified();
    this->mesh->Initialize();
    this->mesh->SetPoints(this->points);
    this->mesh->SetPolys(this->triangles);
}

void Surface::read_obj(const string &basicString) {
    //std::cout << basicString << std::endl;
    auto reader = vtkSmartPointer<vtkOBJReader>::New();
    reader->SetFileName(basicString.c_str());
    reader->Update();
    vtkSmartPointer<vtkPolyData> origpd = reader->GetOutput();
    auto points = vtkSmartPointer<vtkPoints>::New();
    auto polys = vtkSmartPointer<vtkCellArray>::New();
    auto newpd = vtkSmartPointer<vtkPolyData>::New();
    points->DeepCopy(origpd->GetPoints());
    polys->DeepCopy(origpd->GetPolys());
    newpd->SetPoints(points);
    newpd->SetPolys(polys);

    this->mesh = newpd;
    this->points = points;
    this->triangles = polys;
    //other init
    this->vec_tri = this->getTrianglesAsVec();

    this->compute_tri_neigbours();
    this->compute_points_neigbours();

}
#define EPSILON 0.000001
#define BEPSILON (0.000001 * 10)

/* this edge to edge test is based on Franlin Antonio's gem:
   "Faster Line Segment Intersection", in Graphics Gems III,
   pp. 199-202 */
#define EDGE_EDGE_TEST(V0, U0, U1)                                  \
  Bx = U0[i0] - U1[i0];                                             \
  By = U0[i1] - U1[i1];                                             \
  Cx = V0[i0] - U0[i0];                                             \
  Cy = V0[i1] - U0[i1];                                             \
  f = Ay * Bx - Ax * By;                                            \
  d = By * Cx - Bx * Cy;                                            \
  if ((f > 0 && d >= 0 && d <= f) || (f < 0 && d <= 0 && d >= f)) { \
    e = Ax * Cy - Ay * Cx;                                          \
    if (f > 0) {                                                    \
      if (e >= 0 && e <= f) return 1;                               \
    }                                                               \
    else {                                                          \
      if (e <= 0 && e >= f) return 1;                               \
    }                                                               \
  }
#define EDGE_AGAINST_TRI_EDGES(V0, V1, U0, U1, U2) \
  {                                                \
    double Ax, Ay, Bx, By, Cx, Cy, e, d, f;        \
    Ax = V1[i0] - V0[i0];                          \
    Ay = V1[i1] - V0[i1];                          \
    /* test edge U0,U1 against V0,V1 */            \
    EDGE_EDGE_TEST(V0, U0, U1);                    \
    /* test edge U1,U2 against V0,V1 */            \
    EDGE_EDGE_TEST(V0, U1, U2);                    \
    /* test edge U2,U1 against V0,V1 */            \
    EDGE_EDGE_TEST(V0, U2, U0);                    \
  }
#define POINT_IN_TRI(V0, U0, U1, U2)          \
  {                                           \
    double a, b, c, d0, d1, d2;               \
    /* is T1 completly inside T2? */          \
    /* check if V0 is inside tri(U0,U1,U2) */ \
    a = U1[i1] - U0[i1];                      \
    b = -(U1[i0] - U0[i0]);                   \
    c = -a * U0[i0] - b * U0[i1];             \
    d0 = a * V0[i0] + b * V0[i1] + c;         \
                                              \
    a = U2[i1] - U1[i1];                      \
    b = -(U2[i0] - U1[i0]);                   \
    c = -a * U1[i0] - b * U1[i1];             \
    d1 = a * V0[i0] + b * V0[i1] + c;         \
                                              \
    a = U0[i1] - U2[i1];                      \
    b = -(U0[i0] - U2[i0]);                   \
    c = -a * U2[i0] - b * U2[i1];             \
    d2 = a * V0[i0] + b * V0[i1] + c;         \
    if (d0 * d1 > 0.0) {                      \
      if (d0 * d2 > 0.0) return 1;            \
    }                                         \
  }

int coplanar_tri_tri(double N[3], double V0[3], double V1[3], double V2[3], double U0[3], double U1[3], double U2[3])
{
    double A[3];
    short i0, i1;
    /* first project onto an axis-aligned plane, that maximizes the area */
    /* of the triangles, compute indices: i0,i1. */
    A[0] = fabs(N[0]);
    A[1] = fabs(N[1]);
    A[2] = fabs(N[2]);
    if (A[0] > A[1]) {
        if (A[0] > A[2]) {
            i0 = 1; /* A[0] is greatest */
            i1 = 2;
        }
        else {
            i0 = 0; /* A[2] is greatest */
            i1 = 1;
        }
    }
    else /* A[0]<=A[1] */
    {
        if (A[2] > A[1]) {
            i0 = 0; /* A[2] is greatest */
            i1 = 1;
        }
        else {
            i0 = 0; /* A[1] is greatest */
            i1 = 2;
        }
    }
    /* test all edges of triangle 1 against the edges of triangle 2 */
    EDGE_AGAINST_TRI_EDGES(V0, V1, U0, U1, U2);
    EDGE_AGAINST_TRI_EDGES(V1, V2, U0, U1, U2);
    EDGE_AGAINST_TRI_EDGES(V2, V0, U0, U1, U2);

    /* finally, test if tri1 is totally contained in tri2 or vice versa */
    POINT_IN_TRI(V0, U0, U1, U2);
    POINT_IN_TRI(U0, V0, V1, V2);

    return 0;
}

#define ISECT(VV0, VV1, VV2, D0, D1, D2, isect0, isect1) \
  isect0 = VV0 + (VV1 - VV0) * D0 / (D0 - D1);           \
  isect1 = VV0 + (VV2 - VV0) * D0 / (D0 - D2);\

#define COMPUTE_INTERVALS(VV0, VV1, VV2, D0, D1, D2, D0D1, D0D2, isect0, isect1,res) \
  if (D0D1 > 0.0f) {                                                             \
    /* here we know that D0D2<=0.0 */                                            \
    /* that is D0, D1 are on the same side, D2 on the other or on the plane */   \
    ISECT(VV2, VV0, VV1, D2, D0, D1, isect0, isect1);                            \
  }                                                                              \
  else if (D0D2 > 0.0f) {                                                        \
    /* here we know that d0d1<=0.0 */                                            \
    ISECT(VV1, VV0, VV2, D1, D0, D2, isect0, isect1);                            \
  }                                                                              \
  else if (D1 * D2 > 0.0f || D0 != 0.0f) {                                       \
    /* here we know that d0d1<=0.0 or that D0!=0.0 */                            \
    ISECT(VV0, VV1, VV2, D0, D1, D2, isect0, isect1);                            \
  }                                                                              \
  else if (D1 != 0.0f) {                                                         \
    ISECT(VV1, VV0, VV2, D1, D0, D2, isect0, isect1);                            \
  }                                                                              \
  else if (D2 != 0.0f) {                                                         \
    ISECT(VV2, VV0, VV1, D2, D0, D1, isect0, isect1);                            \
  }                                                                              \
  else {                                                                         \
    /* triangles are coplanar */                                                 \
    res = coplanar_tri_tri(N1, v0, v1, v2, u0, u1, u2);                         \
  }
/* sort so that a<=b */
#define SORT(a, b) \
  if (a > b) {     \
    double c;      \
    c = a;         \
    a = b;         \
    b = c;         \
  }

bool Surface::intersection_triangles(double *v0, double *v1, double *v2, double *u0, double *u1, double *u2) {
    auto N1 = Point::cross_product(Point::substract(v1, v0), Point::substract(v2, v0));
    auto d1 = - Point::scalar(N1,v0);
    double du0, du1, du2, dv0, dv1, dv2, fdu0, fdu1, fdu2, fdv0, fdv1, fdv2;
    double isect1[2], isect2[2];
    double du0du1, du0du2, dv0dv1, dv0dv2;
    short index;
    double vp0, vp1, vp2;
    double up0, up1, up2;
    double b, c, max;

    //(2)eq
    du0 = Point::scalar(N1, u0) + d1;
    du1 = Point::scalar(N1, u1) + d1;
    du2 = Point::scalar(N1, u2) + d1;
    fdu0 = fabs(du0);
    fdu1 = fabs(du1);
    fdu2 = fabs(du2);


    if ((du0 != 0) && (du1 != 0) && (du2 != 0)
        && ( ((du0 > 0) && (du1 > 0) && (du2 > 0))
        || ((du0 < 0) && (du1 < 0) && (du2 < 0)) ) ) {
        delete N1;
        return false; }

    if (fdu0 < EPSILON) du0 = 0;
    if (fdu1 < EPSILON) du1 = 0;
    if (fdu2 < EPSILON) du2 = 0;

    du0du1 = du0 * du1;
    du0du2 = du0 * du2;

    if ((fdu0 < BEPSILON) && (fdu1 < BEPSILON) && (fdu2 < EPSILON)) du0du1 = du0du2 = du0 = du1 = du2 = 0.0;


    auto N2 = Point::cross_product(Point::substract(u1, u0), Point::substract(u2, u0));
    auto d2 = - Point::scalar(N2, u0);

    dv0 = Point::scalar(N2, v0) + d2;
    dv1 = Point::scalar(N2, v1) + d2;
    dv2 = Point::scalar(N2, v2) + d2;
    fdv0 = fabs(dv0);
    fdv1 = fabs(dv1);
    fdv2 = fabs(dv2);

    dv0dv1 = dv0 * dv1;
    dv0dv2 = dv0 * dv2;

    if (fdv0 < EPSILON) dv0 = 0;
    if (fdv1 < EPSILON) dv1 = 0;
    if (fdv2 < EPSILON) dv2 = 0;
    dv0dv1 = dv0 * dv1;
    dv0dv2 = dv0 * dv2;
    if ((fdv0 < BEPSILON) && (fdv1 < BEPSILON) && (fdv2 < EPSILON)) dv0dv1 = dv0dv2 = dv0 = dv1 = dv2 = 0.0;

    auto D = Point::cross_product(N1,N2);

    /* compute and index to the largest component of D */
    max = fabs(D[0]);
    index = 0;
    b = fabs(D[1]);
    c = fabs(D[2]);
    if (b > max) max = b, index = 1;
    if (c > max) max = c, index = 2;

    /* this is the simplified projection onto L*/
    vp0 = v0[index];
    vp1 = v1[index];
    vp2 = v2[index];

    up0 = u0[index];
    up1 = u1[index];
    up2 = u2[index];


    int  res = -1;
    /* compute interval for triangle 1 */
    COMPUTE_INTERVALS(vp0, vp1, vp2, dv0, dv1, dv2, dv0dv1, dv0dv2, isect1[0], isect1[1],res);
    if (res!=-1) {return res;std::cout << 5 <<std::endl;}
    /* compute interval for triangle 2 */
    COMPUTE_INTERVALS(up0, up1, up2, du0, du1, du2, du0du1, du0du2, isect2[0], isect2[1],res);
    if (res!=-1) {return res;std::cout << 6 <<std::endl;}
    SORT(isect1[0], isect1[1]);
    SORT(isect2[0], isect2[1]);
    if (isect1[1] < isect2[0] || isect2[1] < isect1[0]) return false;
    std::cout << 7 <<std::endl;
    return true;


}

std::vector<std::vector<int>> Surface::getTrianglesAsVec() {

    auto res = std::vector<std::vector<int>>(this->triangles->GetNumberOfCells());
    auto ellist = vtkSmartPointer<vtkIdList>::New();
    for (int i = 0; i < this->triangles->GetNumberOfCells();i++){
        //std::cout << i << std::endl;
        auto tvec = std::vector<int>(3);

        ellist->Initialize();
        this->triangles->GetCellAtId(i,ellist);

        tvec[0]=(ellist->GetId(0));
        tvec[1]=(ellist->GetId(1));
        tvec[2]=(ellist->GetId(2));
        res[i] = tvec;


    }
    return res;
}

std::vector<std::vector<double>> Surface::getPointsAsVec() {
    auto res = std::vector<std::vector<double>>(this->points->GetNumberOfPoints());
    double x,y,z;
    for (int i = 0; i < this->points->GetNumberOfPoints();i++){
        //std::cout << i << std::endl;
        auto tvec = std::vector<double>(3);

        auto pt = this->points->GetPoint(i);
        x = pt[0];
        y = pt[1];
        z = pt[2];
        tvec[0] = x;
        tvec[1] = y;
        tvec[2] = z;

        res[i]= tvec;
        delete pt;

    }
    return res;
}

void Surface::apply_transformation(arma::mat pre_transformation) {

    TransformMatrix matrix = TransformMatrix();
    matrix.setMatrix(pre_transformation);
    this->apply_transformation(matrix);

}
std::shared_ptr<spdlog::logger> Surface::_logger=spdlog::rotating_logger_st("file_logger", "logs/mylogfile", 1048576 * 5, 25);

bool Surface::triangle_intersection(const double* V10, const double* V11, const double* V12, const double* V20, const double* V21, const double* V22 )
{//this is the de
    //calculate triangle plane 2
    //cout<<"test triangles"<<endl;


//    Surface::_logger->error("Entered");



        short eq = 0;
        if (Point::isEqual(V10, V20) || Point::isEqual(V10, V21) || Point::isEqual(V10, V22)) eq++;
        if (Point::isEqual(V11, V20) || Point::isEqual(V11, V21) || Point::isEqual(V11, V22)) eq++;
        if (Point::isEqual(V12, V20) || Point::isEqual(V12, V21) || Point::isEqual(V12, V22)) eq++;

        if (eq == 1) { Surface::_logger->error("Succesful exit in 1");return false;}

        auto v1 =  Point::substract(V21, V20);
        auto v2 = Point::substract(V22, V20);
        double *N2 = Point::cross_product(v1, v2);
        delete v1;
        delete v2;
        double d2 = -Point::scalar(N2, V20);

//	T testV[]={1,2,3};
//	T testV2[]= {4,5,6};
//	T* test=cross_prod<T>(testV,testV2);
        //T* N2=cross_prod<T>(testV,testV2);

        //cout<<"test "<<test[0]<<" "<<test[1]<<" "<<test[2]<<" "<<dot_prod(test,testV2)<<endl;
        //calculate distance of vertices from plane 1
        double dist10 = Point::scalar(N2, V10) + d2;
        double dist11 = Point::scalar(N2, V11) + d2;
        double dist12 = Point::scalar(N2, V12) + d2;
        //cout<<"DIST AGAIN "<<dist10<<" "<<dot_prod(N2,V22)+d2<<" "<<dot_prod(N2,V20)+d2<<" "<<dot_prod(N2,V21)+d2<<endl;
        if (((dist10 >= 0) && (dist11 >= 0) && (dist12 >= 0)) || ((dist10 <= 0) && (dist11 <= 0) && (dist12 <= 0))) {
            delete N2;
//            Surface::_logger->error("Succesful exit in 2");
            return false;//not all points are on same side of plan
        }
//	}else if (( dist10 == dist11 ) && (dist11 == dist12) && ( dist12 == 0 )) //test for coplanar
//	{
//		delete N2;
//		return false;
//	}else

//        if ((dist10 == 0) || (dist11 == 0) || (dist12 == 0)) {
//            delete[] N2;
////            Surface::_logger->error("Succesful exit in 3");
//            return false;
//        } else {
            //calculate intersection line

            auto v3 = Point::substract(V11, V10);
            auto v4 = Point::substract(V12, V10);
            double *N1 = Point::cross_product(v3, v4);
            delete[] v3;
            delete[] v4;
            double d1 = -Point::scalar(N1, V10);

            double dist20 = Point::scalar(N1, V20) + d1;
            double dist21 = Point::scalar(N1, V21) + d1;
            double dist22 = Point::scalar(N1, V22) + d1;

            if (((dist20 >= 0) && (dist21 >= 0) && (dist22 >= 0)) ||
                ((dist20 <= 0) && (dist21 <= 0) && (dist22 <= 0))) {
                delete[] N2;
                delete[] N1;
//                Surface::_logger->error("Succesful exit in 3_1");
                return false;//not all points are on same side of plan
            }
            //claulcate line in=tersection (without intercept)
            double *D = Point::cross_product(N1, N2);
            //calculate max axis to project onto
/*		short max_axis = 0;
		if (abs(D[1])>abs(D[0]))
			max_axis=1;
		else if (abs(D[2])>abs(D[max_axis]))
			max_axis=2;

		delete N1;
		delete N2;
		delete D;

		T p10 = V10[max_axis];
		T p11 = V11[max_axis];
		T p12 = V12[max_axis];

		T p20 = V20[max_axis];
		T p21 = V21[max_axis];
		T p22 = V22[max_axis];
*/
            double p10 = Point::scalar(D, V10);//[max_axis];
            double p11 = Point::scalar(D, V11);
            double p12 = Point::scalar(D, V12);

            double p20 = Point::scalar(D, V20);
            double p21 = Point::scalar(D, V21);
            double p22 = Point::scalar(D, V22);

            if (abs(dist10) < 1e-4) dist10 = 0;
            if (abs(dist11) < 1e-4) dist11 = 0;
            if (abs(dist12) < 1e-4) dist12 = 0;
            if (abs(dist20) < 1e-4) dist20 = 0;
            if (abs(dist21) < 1e-4) dist21 = 0;
            if (abs(dist22) < 1e-4) dist22 = 0;

            	//cout<<"dist1 "<<dist10<<" "<<dist11<<" "<<dist12<<" "<<dist20<<" "<<dist21<<" "<<dist22<<endl;

            if ((dist20 == 0) || (dist21 == 0) || (dist22 == 0)) {
                //cout<<"not intersect 2"<<endl;
                delete[] D;
                delete[] N2;
                delete[] N1;
//                Surface::_logger->error("Succesful exit in 4");
                return false;
            }
            if ((dist10 == dist12) || (dist11 == dist12) || (dist11 == dist10)) {Surface::_logger->error("pr2"); cout << "problem" << endl; }
            double t11, t12, t21, t22; //intersection parameters

            //get triangle 1 interval
            if (((dist10 >= 0) && (dist11 >= 0)) || ((dist10 <= 0) && (dist11 <= 0))) {
                t11 = p10 - (p10 - p12) * (dist10 / (dist10 - dist12));
                t12 = p11 - (p11 - p12) * (dist11 / (dist11 - dist12));
            } else if (((dist10 >= 0) && (dist12 >= 0)) || ((dist10 <= 0) && (dist12 <= 0))) {
                t11 = p10 - (p10 - p11) * (dist10 / (dist10 - dist11));
                t12 = p12 - (p12 - p11) * (dist12 / (dist12 - dist11));
            } else {
                t11 = p11 - (p11 - p10) * (dist11 / (dist11 - dist10));
                t12 = p12 - (p12 - p10) * (dist12 / (dist12 - dist10));
            }
            if ((dist20 == dist21) || (dist20 == dist22) || (dist21 == dist22)) {Surface::_logger->error("pr2"); cout << "problem2" << endl; }
            //get triangle 2 interval
            if (((dist20 >= 0) && (dist21 >= 0)) || ((dist20 <= 0) && (dist21 <= 0))) {
                t21 = p20 - (p20 - p22) * (dist20 / (dist20 - dist22));
                t22 = p21 - (p21 - p22) * (dist21 / (dist21 - dist22));
            } else if (((dist20 >= 0) && (dist22 >= 0)) || ((dist20 <= 0) && (dist22 <= 0))) {
                t21 = p20 - (p20 - p21) * (dist20 / (dist20 - dist21));
                t22 = p22 - (p22 - p21) * (dist22 / (dist22 - dist21));
            } else {
                t21 = p21 - (p21 - p20) * (dist21 / (dist21 - dist20));
                t22 = p22 - (p22 - p20) * (dist22 / (dist22 - dist20));
            }
//		cout<<V10[0]<<" "<<V10[1]<<" "<<V10[2]<<endl;
//		cout<<V11[0]<<" "<<V11[1]<<" "<<V11[2]<<endl;
//		cout<<V12[0]<<" "<<V12[1]<<" "<<V12[2]<<endl;
//		cout<<V20[0]<<" "<<V20[1]<<" "<<V20[2]<<endl;
//		cout<<V21[0]<<" "<<V21[1]<<" "<<V21[2]<<endl;
//		cout<<V22[0]<<" "<<V22[1]<<" "<<V22[2]<<endl;
//		cout<<"dist "<<dist10<<" "<<dist11<<" "<<dist12<<" "<<dist20<<" "<<dist21<<" "<<dist22<<endl;
//		cout<<"intervals "<<t11<<" "<<t12<<" "<<t21<<" "<<t22<<endl;
            //check interval overlap
            if (((t21 <= t11) && (t21 <= t12) && (t22 <= t11) && (t22 <= t12)) || \
             ((t21 >= t11) && (t21 >= t12) && (t22 >= t11) && (t22 >= t12))){
                delete[] D;
                delete[] N1;
                delete[] N2;
//                Surface::_logger->error("Succesful exit in 5");
                return false;
        } else { delete[] D; delete[] N1; delete[] N2;}
        //}
//        Surface::_logger->error("Succesful exit in 6");

        return true;

}

Surface::Surface() {

}

void Surface::triangle_normalisation(int iterations,double fraction) {

    int pt1,pt2,pt3;



    for (int i = 0; i < iterations;i++){

        for (int pt = 0; pt< this->points->GetNumberOfPoints();pt++){

            std::vector<double> areas;
            //compute areas
            int tr_size = this->point_tri[pt].size();
            for (int tri = 0;tri<tr_size;tri++){
                int triangle = this->point_tri[pt][tri];

                pt1 = this->vec_tri[triangle][0];
                pt2 = this->vec_tri[triangle][1];
                pt3 = this->vec_tri[triangle][2];

                Point pt_1 = Point(this->points->GetPoint(pt1));
                Point pt_2= Point(this->points->GetPoint(pt2));
                Point pt_3= Point(this->points->GetPoint(pt3));

                Point v1 = pt_2 - pt_1;
                Point v2 = pt_3 - pt_1;

                areas.push_back(abs(Point::dot(v1,v2)));

            }


            //compute maximum
            int max_i = 0;
            for (int mx = 1;mx < areas.size();mx++){
                if( areas[mx]>areas[max_i]){
                    max_i = mx;
                }
            }
            int max_tri = this->point_tri[pt][max_i];
            //compute points
            int ind_p,ind_o1,ind_o2;
            ind_p = pt;
            if (this->vec_tri[max_tri][0] == pt){

                ind_o1 = this->vec_tri[max_tri][1];
                ind_o2 = this->vec_tri[max_tri][2];
            }else{
                if (this->vec_tri[max_tri][1] == pt){
                    ind_o1 = this->vec_tri[max_tri][0];
                    ind_o2 = this->vec_tri[max_tri][2];
                }
                else{
                    ind_o1 = this->vec_tri[max_tri][0];
                    ind_o2 = this->vec_tri[max_tri][1];
                }
            }
            Point point =  Point(this->getPoint(pt));

            Point a1 = Point(this->getPoint(ind_o1));
            Point a2 = Point(this->getPoint(ind_o2));
            Point mid = a1 + a2;
            mid = mid/2;

            Point dir = point - mid;

            dir = dir * fraction;

            point = point-dir;

            this->points->SetPoint(pt,point.getPt());


        }



    }

    this->update_mesh();
}

void Surface::compute_tri_neigbours() {
        auto res =  std::vector<std::set<int>>();
        for (int i = 0; i < this->vec_tri.size();i++){
            auto t_res = std::set<int>();


            for (int j = 0; j < this->vec_tri.size();j++){
                if (j!=i){
                    if ((this->vec_tri[i][0] == this->vec_tri[j][0]) ||
                        (this->vec_tri[i][0] == this->vec_tri[j][1]) ||
                        (this->vec_tri[i][0] == this->vec_tri[j][2]) ||
                        (this->vec_tri[i][1] == this->vec_tri[j][0]) ||
                        (this->vec_tri[i][1] == this->vec_tri[j][1]) ||
                        (this->vec_tri[i][1] == this->vec_tri[j][2]) ||
                        (this->vec_tri[i][2] == this->vec_tri[j][0]) ||
                        (this->vec_tri[i][2] == this->vec_tri[j][1]) ||
                        (this->vec_tri[i][2] == this->vec_tri[j][2])){
                        t_res.insert(j);
                    }
                }

            }
            res.push_back(t_res);
        }

    this->tri_neighb = res;



}

void Surface::compute_points_neigbours() {
    int k,i,j;

    std::vector<vector<int>> point_neigbours;
    for ( i = 0; i < this->points->GetNumberOfPoints();i++){

        std::vector<int> p_neigb;
        for ( j = 0; j < this->vec_tri.size();j++){


            //is i inside points

            for ( k = 0; k< 3;k++){
                if (this->vec_tri[j][k] == i){
                    p_neigb.push_back(j);
                    break;
                }
            }


        }
        point_neigbours.push_back(p_neigb);
    }

    this->point_tri = point_neigbours;
}

double *Surface::getPoint(int pos) {
    return  this->points->GetPoint(pos);

}

void Surface::update_mesh() {
    this->mesh.New();
    this->mesh->SetPoints(this->points);
    this->mesh->SetPolys(this->triangles);
}

void Surface::lab_move_points(VolumeDouble &mask, double threshold) {
    auto normalsGen =  vtkSmartPointer<vtkPolyDataNormals>::New();
    normalsGen->SetInputData(this->mesh);
    normalsGen->ComputeCellNormalsOff();
    normalsGen->ComputePointNormalsOn();
    bool a1 = normalsGen->GetAutoOrientNormals();
    normalsGen->SetAutoOrientNormals(true);
    normalsGen->Update();
    auto normals = normalsGen->GetOutput();
    auto normals4= normals->GetPointData()->GetNormals();


    double nm;
    for(int i = 0; i < this->points->GetNumberOfPoints(); i++){
        auto vox = Point(this->points->GetPoint(i));
        double* normal = normals4->GetTuple(i);


        auto norm2 = Point(normal);
        Point res_vox= vox.move_in_value_dir(mask,norm2,0.1,threshold);

        this->points->SetPoint(i,res_vox.getPt());
        //cout<<  "Process point :"<< i << endl;
    }

    this->mesh->Initialize();
    this->mesh->SetPoints(this->points);
    this->mesh->SetPolys(this->triangles);
}

std::vector<std::vector<std::vector<double>>> Surface::calculate_normals(double mm, int npts) {


    double dt = (2 * mm) / (npts -1);
    auto norm_calc = vtkSmartPointer<vtkPolyDataNormals>::New();
    norm_calc->SetInputData(this->mesh);
    norm_calc->ComputeCellNormalsOff();
    norm_calc->ComputePointNormalsOn();
    norm_calc->SetAutoOrientNormals(true);

    norm_calc->Update();
    auto normals = norm_calc->GetOutput()->GetPointData()->GetNormals();
    auto res = std::vector<std::vector<std::vector<double>>>();

    for (int i = 0; i < this->points->GetNumberOfPoints();i++){
        auto tmp = std::vector<std::vector<double>>();
        auto norm = normals->GetTuple(i);


        auto pt = this->points->GetPoint(i);


        for (int j = 0; j < npts;j++){
            auto res_pt = std::vector<double>();

            res_pt.push_back( pt[0] + norm[0]*(-mm + j*dt));
            res_pt.push_back( pt[1] + norm[1]*(-mm + j*dt));
            res_pt.push_back( pt[2] + norm[2]*(-mm + j*dt));

            tmp.push_back(res_pt);

        }
        res.push_back(tmp);
    }

    return res;

}

void Surface::saveImage(const std::string filename) {

    // Visualize
    vtkSmartPointer<vtkPolyDataMapper> mapper =
            vtkSmartPointer<vtkPolyDataMapper>::New();
    mapper->SetInputData(this->mesh);

    vtkSmartPointer<vtkActor> actor =
            vtkSmartPointer<vtkActor>::New();
    actor->SetMapper(mapper);

    vtkSmartPointer<vtkRenderer> renderer =
            vtkSmartPointer<vtkRenderer>::New();
    vtkSmartPointer<vtkRenderWindow> renderWindow =
            vtkSmartPointer<vtkRenderWindow>::New();
    renderWindow->AddRenderer(renderer);
    renderWindow->OffScreenRenderingOn();
    renderWindow->SetAlphaBitPlanes(1); //enable usage of alpha channel

    vtkSmartPointer<vtkRenderWindowInteractor> renderWindowInteractor =
            vtkSmartPointer<vtkRenderWindowInteractor>::New();
    renderWindowInteractor->SetRenderWindow(renderWindow);

    renderer->AddActor(actor);
    renderer->SetBackground(1,1,1); // Background color white
    //renderer->Render();
    renderWindow->Render();

    // Screenshot
    vtkSmartPointer<vtkWindowToImageFilter> windowToImageFilter =
            vtkSmartPointer<vtkWindowToImageFilter>::New();
    windowToImageFilter->SetInput(renderWindow);
    windowToImageFilter->SetScale(3); //set the resolution of the output image (3 times the current resolution of vtk render window)
    windowToImageFilter->SetInputBufferTypeToRGBA(); //also record the alpha (transparency) channel
    windowToImageFilter->ReadFrontBufferOff(); // read from the back buffer
    windowToImageFilter->Update();

    vtkSmartPointer<vtkPNGWriter> writer =
            vtkSmartPointer<vtkPNGWriter>::New();
    writer->SetFileName(filename.c_str());
    writer->SetInputConnection(windowToImageFilter->GetOutputPort());
    writer->Write();

}

bool RayIntersectsTriangle(Point rayOrigin,
                           Point rayVector,
                           std::vector<Point>& inTriangle,
                           Point& outIntersectionPoint)
{
    //const float EPSdILON = 0.0000001;
    Point vertex0 = inTriangle[0];
    Point vertex1 = inTriangle[1];
    Point vertex2 = inTriangle[2];
    //std::cout << "v2 " << vertex0.getX() << std::endl;
    Point edge1, edge2, h, s, q;
    double a,f,u,v;
    edge1 = vertex1 - vertex0;
    edge2 = vertex2 - vertex0;
    h = Point::cross_product(rayVector,edge2);
    a = Point::scalar(edge1,h);
    if (a > -EPSILON && a < EPSILON){
        //std::cout << "TRUE 1" << std::endl;
        //std::cout << "v2 " << vertex0.getX() << std::endl;
        return false;    // This ray is parallel to this triangle.
    }

    f = 1.0/a;
    s = rayOrigin - vertex0;
    u = f * Point::scalar(s,h);
    if (u < 0.0 || u > 1.0) {
        //std::cout << "TRUE 2" << std::endl;
        //std::cout << "v2 " << vertex0.getX() << std::endl;
        return false; }
    q = Point::cross_product(s,edge1);
    v = f * Point::scalar(rayVector,q);
    if (v < 0.0 || u + v > 1.0)
        return false;
    // At this stage we can compute t to find out where the intersection point is on the line.
    double t = f * Point::scalar(edge2,q);
    if (t > EPSILON) // ray intersection
    {
        outIntersectionPoint = rayOrigin + rayVector * t;
        return true;
    }
    else // This means that there is a line intersection but not a ray intersection.
    {
        //std::cout << "TRUE 3" << std::endl;
        return false; }
}

double SignedVolume(Point a, Point b,Point c,Point d)
{
    Point diff = Point::cross_product(b-a,c-a);
    return (1/6.0) * Point::scalar(diff ,d-a);
}
bool isIntersectsTriangle(Point rayOrigin,
                           Point rayVector,
                           std::vector<Point>& inTriangle){
    Point p1 = inTriangle[0];
    Point p2 = inTriangle[1];
    Point p3 = inTriangle[2];

    if (
            ((SignedVolume(rayOrigin,p1,p2,p3) < 0) & (SignedVolume(rayVector,p1,p2,p3) > 0)) |
                    ((SignedVolume(rayOrigin,p1,p2,p3) > 0) & (SignedVolume(rayVector,p1,p2,p3) < 0))
            )
    {
        if ( ((SignedVolume(rayOrigin,rayVector,p1,p2) > 0) & (SignedVolume(rayOrigin,rayVector,p2,p3) > 0) & (SignedVolume(rayOrigin,rayVector,p3,p1)>0))
            | (SignedVolume(rayOrigin,rayVector,p1,p2) < 0) & (SignedVolume(rayOrigin,rayVector,p2,p3) < 0) & (SignedVolume(rayOrigin,rayVector,p3,p1)<0)
        ) {
            return true;
        } else return false;

    } else
    {
        return false;
    }
}


Point intersectionLineTriangle(Point q1,
                               Point q2,
                               std::vector<Point>& inTriangle){
    Point p1 = inTriangle[0];
    Point p2 = inTriangle[1];
    Point p3 = inTriangle[2];

    Point N = Point::cross_product(p2-p1, p3-p1);
    double t = - (Point::dot(q1-p1,N))/Point::dot(q2-q1,N);

    return q1 + (q2-q1)*t;
}
std::vector<std::vector<double>> Surface::rayMeshIntersection(std::vector<std::vector<double>> start_end)
{
    Point start = Point(start_end[0]);
    Point end = Point(start_end[1]);
    Point ss1,ss2;
    int cnt = 0;
    for (int i = 0; i < this->triangles->GetNumberOfCells();i++){

        auto triangle = this->get_triangle(i);
        Point t_res;
        if (isIntersectsTriangle(start,end,triangle)){

            //std::cout << "true" << std::endl;
            if (ss1.getX() == 0) ss1 = intersectionLineTriangle(start,end,triangle);
            else ss2 = intersectionLineTriangle(start,end,triangle);
            cnt++;

        }




    }



    auto res = std::vector<std::vector<double>>();

    if ((start - ss1).normSquare() < (start - ss2).normSquare() ){
        res.push_back(ss1.toVector());
        res.push_back(ss2.toVector());
    } else
    {
        res.push_back(ss2.toVector());
        res.push_back(ss1.toVector());
    }
    return res;

}




std::vector<Point> Surface::get_triangle(int id) {

    auto res = std::vector<Point>();

    //auto res = std::vector<std::vector<int>>(this->triangles->GetNumberOfCells());
    auto ellist = vtkSmartPointer<vtkIdList>::New();

    auto tvec = std::vector<int>(3);

    ellist->Initialize();
    this->triangles->GetCellAtId(id,ellist);

    tvec[0]=(ellist->GetId(0));

    res.emplace_back(this->getPoint(tvec[0]));
    tvec[1]=(ellist->GetId(1));
    res.emplace_back(Point(this->getPoint(tvec[1])));
    tvec[2]=(ellist->GetId(2));
    res.emplace_back( Point(this->getPoint(tvec[2])));
    return res;


}

std::vector<std::vector<double>> Surface::getTriangleCenters() {
    auto filter = vtkSmartPointer<vtkCellCenters>::New();
    filter->SetInputData( this->mesh);
    filter->VertexCellsOn();
    filter->Update();
    auto ret = std::vector<std::vector<double>>();
    // Access the cell centers
    for (vtkIdType i = 0; i < filter->GetOutput()->GetNumberOfPoints();
         i++)
    {
        std::vector<double> vec1;
        double p[3];
        filter->GetOutput()->GetPoint(i, p);
        vec1.push_back(p[0]);vec1.push_back(p[1]);vec1.push_back(p[2]);
        ret.push_back(vec1);
    }
    return ret;

}

std::vector<int> Surface::rayMeshInterInd(std::vector<std::vector<double>> start_end) {
    Point start = Point(start_end[0]);
    Point end = Point(start_end[1]);
    Point ss1,ss2;
    int res1 = -1;
    int res2 = -1;
    int cnt = 0;
    for (int i = 0; i < this->triangles->GetNumberOfCells();i++){

        auto triangle = this->get_triangle(i);
        Point t_res;
        if (isIntersectsTriangle(start,end,triangle)){

            //std::cout << "true" << std::endl;


            if (ss1.getX() == 0) {
                res1 = i;
                ss1 = intersectionLineTriangle(start, end, triangle); }
            else { ss2 = intersectionLineTriangle(start, end, triangle);
                res2 = i;
            }
            cnt++;

        }




    }



    auto res = std::vector<int>();

    if ((start - ss1).normSquare() < (start - ss2).normSquare() ){
        res.push_back(res1);
        res.push_back(res2);
    } else
    {
        res.push_back(res2);
        res.push_back(res1);
    }
    return res;


}

Surface::Surface(const Surface &surface) {
    this->mesh = vtkSmartPointer<vtkPolyData>::New();
    this->mesh->AllocateCopy(surface.mesh);
    this->mesh->DeepCopy(surface.mesh);

    this->points = vtkSmartPointer<vtkPoints>::New();
    this->points->DeepCopy(surface.points);

    this->triangles = vtkSmartPointer<vtkCellArray>::New();
    this->triangles->DeepCopy(surface.triangles);

    std::copy(surface.tri_neighb.begin(),surface.tri_neighb.end(), std::back_inserter(this->tri_neighb));
    std::copy(surface.point_tri.begin(),surface.point_tri.end(), std::back_inserter(this->point_tri));
    std::copy(surface.vec_tri.begin(),surface.vec_tri.end(), std::back_inserter(this->vec_tri));


}

double Surface::distanceToPoint(double x, double y, double z) {
    vtkSmartPointer<vtkImplicitPolyDataDistance> ipd = vtkSmartPointer<vtkImplicitPolyDataDistance>::New();
    ipd->SetInput(this->mesh);
    return ipd->EvaluateFunction(x,y,z);

}
