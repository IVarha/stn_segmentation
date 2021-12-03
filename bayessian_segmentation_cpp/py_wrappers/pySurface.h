//
// Created by ivarh on 11/12/2020.
//

#ifndef BAYESSIAN_SEGMENTATION_CPP_PYSURFACE_H
#define BAYESSIAN_SEGMENTATION_CPP_PYSURFACE_H
#include <string>
#include "Surface.h"
#include "iostream"
#include "triangleintersects.h"
#include <set>
class pySurface {
    std::string name;
    std::vector<std::vector<int>> triangles;
    std::vector<std::set<int>> neighb_tri;

    vtkSmartPointer<vtkPoints> points;
    Surface* mesh = nullptr;
    int p_size = 0;
public:
    explicit pySurface(const std::string &name) : name(name) {
        this->mesh = new Surface();
        this->mesh->read_obj(name);
        //std::cout << 1 << std::endl;
        this->triangles = this->mesh->getTrianglesAsVec();
        //std::cout << 2 << std::endl;
        this->points = this->mesh->getPoints();
        //std::cout << 3 << std::endl;
        this->neighb_tri = compute_neighbours();
    }
    /**
     * Initialisation of python instance of  surface from @see Surface
     *
     *
     *
     *
     *
     * */
    explicit pySurface(const Surface& surf);


    const std::string &getName() const { return name; }

    /**
     * Modifies instance of Surface by replacing vertices
     *
     * */
    void modify_points(std::vector<double> points);

    bool self_intersection_test(const std::vector<double>& new_points);
    /**
     * Applies transformation 'arr' to points
     * @param arr transformation matrix 4x4
     */
    void apply_transformation(const std::vector<std::vector<double>>& arr);

    std::vector<std::vector<int>> getFaces();

    static bool triangles_intersected( std::vector<std::vector<double>> points);
    std::vector<std::set<int>> compute_neighbours();

    void set_image(std::string file_name);

    std::vector<std::vector<std::vector<double>>> generateNormals(double mm_len, int npts);

    /**
     * Calculate points inside mesh
     * @param discretisation
     * @return
     */
    std::vector<std::vector<double>> getInsideMeshPoints( int discretisation);

    /**
     * Calculates points inside of parallelepiped containing mesh
     * @param discretisation
     * @return
     */
    std::vector<std::vector<double>> getInsideBoundaryPoints( int discretisation);

    std::vector<bool> isPointsInside( std::vector<std::vector<double>> points);
    void saveObj(std::string filename);


    std::vector<std::vector<double>> rayTriangleIntersection(std::vector<std::vector<double>> start_end);

    std::vector<std::vector<double>> centresOfTriangles();

    std::vector<int> rayTriangleIntersectionIndexes( std::vector<std::vector<double>> start_end );

    double distanceToPoint(double x, double y, double z){
        return this->mesh->distanceToPoint(x,y,z);
    }

    double computeVolume();



    std::vector<double> getUnpackedCords();
    virtual ~pySurface() {
        if (mesh!= nullptr) delete mesh;
    }

    /**
     *
     * @param label_name path + filename of labels
     * @param transformation transformation from native to mni
     * @param num_iterations
     * @param num_subdivisions
     * @param mapped_label_indices
     * @param fraction
     * @param smooth_numb1
     * @param smooth_numb2
     * @return
     */
    vector<pySurface>
    calculate_labels(string label_name, vector<vector<double>> transformation, int num_iterations, int num_subdivisions,
                     vector<vector<int>> mapped_label_indices,//same labels e.g 1,2 RN, 3,4 -STN
                     double fraction,unsigned int smooth_numb1, unsigned int smooth_numb2);
};


#endif //BAYESSIAN_SEGMENTATION_CPP_PYSURFACE_H
