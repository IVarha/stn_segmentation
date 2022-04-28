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

std::vector<std::vector<double>> pySurface::rayTriangleIntersection(std::vector<std::vector<double>> start_end) {

    return this->mesh->rayMeshIntersection(start_end);
}

std::vector<std::vector<double>> pySurface::centresOfTriangles() {
    return mesh->getTriangleCenters();
}

std::vector<int> pySurface::rayTriangleIntersectionIndexes(std::vector<std::vector<double>> start_end) {

    return this->mesh->rayMeshInterInd(start_end);
}

pySurface::pySurface(const Surface& surface) {

    this->mesh = new Surface(surface);
    this->triangles = this->mesh->getTrianglesAsVec();
    //std::cout << 2 << std::endl;
    this->points = this->mesh->getPoints();
    this->neighb_tri = compute_neighbours();

}
/**
 * Generate a mesh from parameters
 * @param mask MRI label image
 * @param image MRI image
 * @param transformation transformation which we apply to mni space for augmentation
 * @param targ_label_value integer number of label(in image) which we processing e.g. STN = 3 so STN voxels in image w
 * would be marked as 3
 * @param num_subdivisions number of subdivisions number which depends on a number of a vertex (better to use <4) see @relatedalso Surface::generate_sphere
 * @param fraction value which should be reached for a label considered as a correct @refitem Surface::shrink_sphere
 * @param num_iterations number of iterations of smoothing + moving of vertices
 * @param smooth_numb1 number of smoothing iterations see VTK in cycle
 * @param smooth_numb2 final number of smooting iterations (see VTK)
 * @param is_mirror is label mirrored
 * @return mesh from parameters
 */
pySurface generate_mesh(VolumeInt* mask,NiftiImage& image,
                        TransformMatrix& transformation,int targ_label_value,
                        int num_subdivisions,double fraction,int num_iterations
                        , unsigned int smooth_numb1, unsigned int smooth_numb2, bool is_mirror //todo add mirror
                         ){

    auto label_mask = mask->label_to_mask(targ_label_value);


    auto inv_transform = transformation.get_inverse();//from augmented mni to world

    Point centr_of_label = label_mask.center_of_mass();
    //point of center in world coordinate
    auto swap_centre = image.get_voxel_to_world().apply_transform(centr_of_label.getX()
            ,centr_of_label.getY(),centr_of_label.getZ());

    auto lab_centre_mni = transformation.apply_transform(swap_centre); //map labels centre to mni space

    //GENERATE SPHERE IN MNI COORDS!!!

    tuple<double,double,double> ride = {lab_centre_mni[0],lab_centre_mni[1],lab_centre_mni[2]};
    auto sphr = Surface::generate_sphere(100,ride,num_subdivisions);

    sphr.apply_transformation(inv_transform);

    auto W_V_trans= image.get_world_to_voxel();
    sphr.apply_transformation(W_V_trans);

    auto mask1 = label_mask.int_to_double();




    //sphr.write_obj(workdir + "/" + std::to_string(label.first) + "_cent_sphere.obj");
    //SHRINK SPHERE
    sphr.shrink_sphere(mask1,centr_of_label.to_tuple(), fraction);
//    sphr.smoothMesh();

    for (int i = 0;i < num_iterations;i++) {
        //sphr.triangle_normalisation(1, 0.1);
        sphr.smoothMesh(smooth_numb1);
        sphr.lab_move_points(mask1, fraction);
    }
    sphr.smoothMesh(smooth_numb2);
    auto trans = image.get_voxel_to_world();
    sphr.apply_transformation(trans);

    return pySurface(sphr);
}



std::vector<pySurface> pySurface::calculate_labels(std::string label_name, std::vector<std::vector<double>> transformation,//transformation to mni
                                                   int num_iterations, int num_subdivisions,
                            std::vector<std::vector<int>> mapped_label_indices, //same labels e.g 1,2 RN, 3,4 -STN
                            double fraction,unsigned int smooth_numb1, unsigned int smooth_numb2) {

    NiftiImage image = NiftiImage();
    image.read_nifti_image(label_name);
    //transformation matrix
    TransformMatrix tm = TransformMatrix();
    tm.setMatrix(transformation);

    auto* lab1_vol = (VolumeInt*)image.returnImage();

    std::vector<pySurface> res;
    for (auto & mapped_label_indice : mapped_label_indices){
        bool mark = false;
        for (int ind : mapped_label_indice){
                res.push_back(generate_mesh(lab1_vol,image,tm,ind
                        ,num_subdivisions,fraction,num_iterations,smooth_numb1,smooth_numb2, mark )  );
                mark = true;
        }


    }

    return res;




}





pySurface pySurface::calculate_label( vector<vector<vector<bool>>> mask,
                           const vector<vector<double>>& to_mni, //from voxel to MNI
                           int num_iterations, int num_subdivisions, double fraction,
                           unsigned int smooth_numb1, unsigned int smooth_numb2){

    Point center_of_label = VolumeInt::center_of_mass(mask);
    auto transf = TransformMatrix();
    transf.setMatrix(to_mni);

    auto inv_transform = transf.get_inverse();//from augmented mni to world

    //point of center in mni coordinates
    auto swap_centre = transf.apply_transform(center_of_label.getX()
            ,center_of_label.getY(),center_of_label.getZ());

    //GENERATE SPHERE IN MNI COORDS!!!

    tuple<double,double,double> ride = {swap_centre[0],swap_centre[1],swap_centre[2]};
    auto sphr = Surface::generate_sphere(100,ride,num_subdivisions);

    sphr.apply_transformation(inv_transform);


    auto maskV = VolumeInt::mask_to_double(mask);
    sphr.shrink_sphere(maskV,center_of_label.to_tuple(), fraction);
//    sphr.smoothMesh();

    for (int i = 0;i < num_iterations;i++) {
        //sphr.triangle_normalisation(1, 0.1);
        sphr.smoothMesh(smooth_numb1);
        sphr.lab_move_points(maskV, fraction);
    }
    sphr.smoothMesh(smooth_numb2);
    return pySurface(sphr);



}


