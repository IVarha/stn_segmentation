//
// Created by ivarh on 11/12/2020.
//

#include "pySurface.h"
#include <armadillo>

#include <vtkSelectEnclosedPoints.h>
void pySurface::modify_points(std::vector<double> points) {
    this->mesh->apply_points(points);
    this->points = this->mesh->getPoints();
}

bool pySurface::self_intersection_test(const std::vector<double>& new_points) {
    auto tri = this->mesh->getTriangles();
    int sj = 1;

    double V0[3];
    double V1[3];
    double V2[3];
    double U0[3];
    double U1[3];
    double U2[3];
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
        int cnt = 0;
        for (int j = sj; j<tri->GetNumberOfCells();j++){
            bool mrk = false;
            for (auto s_e= this->neighb_tri[i].begin(); s_e!=this->neighb_tri[i].end(); ++s_e)
            {
                if((*s_e)==j)
                {
                    mrk = true;
                    break;
                }
            }
            if (mrk) continue;
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
            if (moeller::TriangleIntersects<double[3]>::triangle(V0,V1,V2,U0,U1,U2))
            //moeller::TriangleIntersects<double[3]>::triangle(V0,V1,V2,U0,U1,U2);
            //if (Surface::triangle_intersection(V0,V1,V2,U0,U1,U2))
                {
//                delete[] V0;
//                delete[] V1;
//                delete[] V2;
//                delete[] U0;
//                delete[] U1;
//                delete[] U2;
                std::cout << "intersected: " << i << " " << j << std::endl;
                return true;

            }


        }
        sj++;
    }
//    delete[] V0;
//    delete[] V1;
//    delete[] V2;
//    delete[] U0;
//    delete[] U1;
//    delete[] U2;

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
    //a.print(std::cout);

    this->mesh->apply_transformation(a);
    this->points = this->mesh->getPoints();


}

bool pySurface::triangles_intersected(std::vector<std::vector<double>> points) {

    double* v1 = new double(3);
    v1[0] = points[0][0];
    v1[1] = points[0][1];
    v1[2] = points[0][2];
    cout << v1[0] << " " << v1[1] << " " << v1[2] <<endl;

    double* v2 = new double(3);
    v2[0] = points[1][0];
    v2[1] = points[1][1];
    v2[2] = points[1][2];
    cout << v2[0] << " " << v2[1] << " " << v2[2] <<endl;
    double* v3 = new double(3);
    v3[0] = points[2][0];
    v3[1] = points[2][1];
    v3[2] = points[2][2];
    cout << v3[0] << " " << v3[1] << " " << v3[2] <<endl;
    double* u1 = new double(3);
    u1[0] = points[3][0];
    u1[1] = points[3][1];
    u1[2] = points[3][2];
    cout << u1[0] << " " << u1[1] << " " << u1[2] <<endl;
    double* u2 = new double(3);
    u2[0] = points[4][0];
    u2[1] = points[4][1];
    u2[2] = points[4][2];
    cout << u2[0] << " " << u2[1] << " " << u2[2] <<endl;
    double* u3 = new double(3);
    u3[0] = points[5][0];
    u3[1] = points[5][1];
    u3[2] = points[5][2];
    cout << u3[0] << " " << u3[1] << " " << u3[2] <<endl;
    bool res = Surface::triangle_intersection(v1,v2,v3,u1,u2,u3);
    delete[] v1,v2,v3,u1,u2,u3;
    return res;
}

std::vector<std::set<int>> pySurface::compute_neighbours() {
    auto res =  std::vector<std::set<int>>();
    for (int i = 0; i < this->triangles.size();i++){
        auto t_res = std::set<int>();


        for (int j = 0; j < this->triangles.size();j++){
            if (j!=i){
                if ((this->triangles[i][0] == this->triangles[j][0]) ||
                        (this->triangles[i][0] == this->triangles[j][1]) ||
                        (this->triangles[i][0] == this->triangles[j][2]) ||
                        (this->triangles[i][1] == this->triangles[j][0]) ||
                        (this->triangles[i][1] == this->triangles[j][1]) ||
                        (this->triangles[i][1] == this->triangles[j][2]) ||
                        (this->triangles[i][2] == this->triangles[j][0]) ||
                        (this->triangles[i][2] == this->triangles[j][1]) ||
                        (this->triangles[i][2] == this->triangles[j][2])){
                    t_res.insert(j);
                }
            }

        }
        res.push_back(t_res);
    }

    return res;
}

void pySurface::set_image(std::string file_name) {



}

std::vector<std::vector<std::vector<double>>> pySurface::generateNormals(double mm_len, int npts) {
    return this->mesh->calculate_normals(mm_len,npts);

}

std::vector<std::vector<double>> pySurface::getInsideMeshPoints(int discretisation) {

    //generate range
    double max_x = -100000;
    double max_y = -100000;
    double max_z = -100000;
    double min_x = 100000;
    double min_y = 100000;
    double min_z = 100000;
    for (int i = 0; i < this->points->GetNumberOfPoints();i++){

        auto pt = this->points->GetPoint(i);

        if (max_x < pt[0]) max_x = pt[0];
        if (min_x > pt[0]) min_x = pt[0];

        if (max_y < pt[1]) max_y = pt[1];
        if (min_y > pt[1]) min_y = pt[1];

        if (max_z < pt[2]) max_z = pt[2];
        if (min_z > pt[2]) min_z = pt[2];
        //std::cout << pt[1] <<std::endl;

    }


    //std::cout << max_x <<std::endl;
    //std::cout << min_x <<std::endl;
    //std::cout << max_y <<std::endl;
    //std::cout << min_y <<std::endl;
    //std::cout << max_z <<std::endl;
    //std::cout << min_z <<std::endl;

    double dx = (max_x - min_x) / (discretisation -1);
    double dy = (max_y - min_y) / (discretisation -1);
    double dz = (max_z - min_z) / (discretisation -1);
    int i,j,k;
    auto pts = vtkSmartPointer<vtkPoints>::New();
    for ( i =0; i< discretisation;i++){
        for ( j =0; j< discretisation;j++){
            for ( k =0; k< discretisation;k++){
                pts->InsertNextPoint( min_x + i*dx, min_y + j*dy, min_z+k*dz);
            }
        }
    }
    auto poly_data = vtkSmartPointer<vtkPolyData>::New();
    poly_data->SetPoints(pts);
    auto encl_points = vtkSmartPointer<vtkSelectEnclosedPoints>::New();

    encl_points->SetInputData(poly_data);
    encl_points->SetSurfaceData(this->mesh->getMesh());
    encl_points->Update();


    auto res = std::vector<std::vector<double>>();
    for (i = 0;i < pts->GetNumberOfPoints();i++){
        if (encl_points->IsInside(i)){
            auto tmp = std::vector<double>();
            auto pt = pts->GetPoint(i);
            tmp.push_back(pt[0]);
            tmp.push_back(pt[1]);
            tmp.push_back(pt[2]);
            res.push_back(tmp);
        }

    }

    return res;
}

std::vector<double> pySurface::getUnpackedCords() {


    auto res = std::vector<double>();
    for (int i=0;i< this->points->GetNumberOfPoints();i++){
        double* ar = this->points->GetPoint(i);

        res.push_back(ar[0]);
        res.push_back(ar[1]);
        res.push_back(ar[2]);

    }
    return res;
}

void pySurface::saveObj(std::string filename) {
    this->mesh->write_obj(filename);
}

double pySurface::computeVolume() {
    return this->mesh->calculate_volume();
}

std::vector<std::vector<double>> pySurface::getInsideBoundaryPoints(int discretisation) {

    //generate range
    double max_x = -100000;
    double max_y = -100000;
    double max_z = -100000;
    double min_x = 100000;
    double min_y = 100000;
    double min_z = 100000;
    for (int i = 0; i < this->points->GetNumberOfPoints();i++){

        auto pt = this->points->GetPoint(i);

        if (max_x < pt[0]) max_x = pt[0];
        if (min_x > pt[0]) min_x = pt[0];

        if (max_y < pt[1]) max_y = pt[1];
        if (min_y > pt[1]) min_y = pt[1];

        if (max_z < pt[2]) max_z = pt[2];
        if (min_z > pt[2]) min_z = pt[2];
        //std::cout << pt[1] <<std::endl;

    }


    double dx = (max_x - min_x) / (discretisation -1);
    double dy = (max_y - min_y) / (discretisation -1);
    double dz = (max_z - min_z) / (discretisation -1);
    int i,j,k;
    auto res = std::vector<std::vector<double>>();
    for ( i =0; i< discretisation;i++){
        for ( j =0; j< discretisation;j++){
            for ( k =0; k< discretisation;k++){
                auto tmp = std::vector<double>();
                tmp.push_back(min_x + i*dx);
                tmp.push_back(min_y + j*dy);
                tmp.push_back(min_z+k*dz);
                res.push_back(tmp);
            }
        }
    }
    return res;
}

std::vector<bool> pySurface::isPointsInside(std::vector<std::vector<double>> points) {
    auto pts = vtkSmartPointer<vtkPoints>::New();
    for (auto it : points){
        pts->InsertNextPoint(it[0],it[1],it[2]);
    }


    auto poly_data = vtkSmartPointer<vtkPolyData>::New();
    poly_data->SetPoints(pts);
    auto encl_points = vtkSmartPointer<vtkSelectEnclosedPoints>::New();
    encl_points->SetInputData(poly_data);
    encl_points->SetSurfaceData(this->mesh->getMesh());
    encl_points->Update();

    auto res = std::vector<bool>();
    for (int i = 0;i < pts->GetNumberOfPoints();i++){
        if (encl_points->IsInside(i)){
            res.push_back(true);
        } else{
            res.push_back(false);
        }

    }
    return res;
}

std::vector<std::vector<int>> pySurface::getFaces() {
    return triangles;
}


