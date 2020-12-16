//
// Created by ivarh on 11/12/2020.
//

#include "pySurface.h"
#include <armadillo>
void pySurface::modify_points(std::vector<double> points) {
    this->mesh->apply_points(points);
}

bool pySurface::self_intersection_test(const std::vector<double>& new_points) {
    auto tri = this->mesh->getTriangles();
    int sj = 1;

    double * V0 = new double[3];
    double* V1 = new double[3];
    double* V2 = new double[3];
    double* U0 = new double[3];
    double* U1 = new double[3];
    double* U2 = new double[3];
    int v0i,v1i,v2i,u0i,u1i,u2i;
    for (int i = 0; i < tri->GetNumberOfCells() - 1; i++){

        v0i = this->triangles.at(i).at(0);
        v1i = this->triangles.at(i).at(1);
        v2i = this->triangles.at(i).at(2);

        V0[0] = new_points[3*v0i];
        V0[1] = new_points[3*v0i+1];
        V0[2] = new_points[3*v0i+2];
        V1[0] = new_points[3*v1i];
        V1[1] = new_points[3*v1i+1];
        V1[2] = new_points[3*v1i+2];
        V2[0] = new_points[3*v2i];
        V2[1] = new_points[3*v2i+1];
        V2[2] = new_points[3*v2i+2];

        for (int j = sj; j<tri->GetNumberOfCells();j++){
            u0i = this->triangles.at(j).at(0);
            u1i = this->triangles.at(j).at(1);
            u2i = this->triangles.at(j).at(2);

            U0[0] = new_points[3*u0i];
            U0[1] = new_points[3*u0i+1];
            U0[2] = new_points[3*u0i+2];
            U1[0] = new_points[3*u1i];
            U1[1] = new_points[3*u1i+1];
            U1[2] = new_points[3*u1i+2];
            U2[0] = new_points[3*u2i];
            U2[1] = new_points[3*u2i+1];
            U2[2] = new_points[3*u2i+2];

            if (Surface::triangle_intersection(V0,V1,V2,U0,U1,U2)){
                delete[] V0;
                delete[] V1;
                delete[] V2;
                delete[] U0;
                delete[] U1;
                delete[] U2;
                std::cout << "intersected: " << i << " " << j << std::endl;
                return true;

            }


        }
        sj++;
    }
    delete[] V0;
    delete[] V1;
    delete[] V2;
    delete[] U0;
    delete[] U1;
    delete[] U2;

    return false;
}

void pySurface::apply_transformation(const std::vector<std::vector<double>>& arr) {



    auto a = arma::Mat<double>(4,4,arma::fill::zeros);
    a(0,0) = arr[0][0];
    a(0,1) = arr[0][1];
    a(0,2) = arr[0][2];
    a(0,3) = arr[0][3];
    a(1,0) = arr[1][0];
    a(1,1) = arr[1][1];
    a(1,2) = arr[1][2];
    a(1,3) = arr[1][3];
    a(2,0) = arr[2][0];
    a(2,1) = arr[2][1];
    a(2,2) = arr[2][2];
    a(2,3) = arr[2][3];
    a(3,0) = arr[3][0];
    a(3,1) = arr[3][1];
    a(3,2) = arr[3][2];
    a(3,3) = arr[3][3];
    a.print(std::cout);

    this->mesh->apply_transformation(a);
    this->points = this->mesh->getPoints();


}


