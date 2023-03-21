#include <iostream>
#include "Surface.h"
#include "CLIParser.h"
#include "NiftiImage.h"
#include "pySurface.h"


#include <vtkNIFTIImageReader.h>
#include <vtkSmartPointer.h>
#include <vtkMatrix4x4.h>
#include <vtkImageData.h>
#include <vtkPointData.h>
#include <vtkDataArray.h>
#include <vtkNIFTIImageHeader.h>
#include <vtkImageBSplineInterpolator.h>
#include <vtkImageBSplineCoefficients.h>
#include <vtkImageSincInterpolator.h>


using namespace std;



unordered_map<int,string> parse_label_descriptor(string fileName){
    ifstream file;
    file.open(fileName,ios::in);
    string line;

    auto result = unordered_map<int,string>();

    if (file.is_open()){

        while (getline(file,line)){
            int pos = line.find(',');

            auto label = line.substr(0,pos);
            int pos2 = line.find(',',pos+1);
            auto mesh = line.substr(pos+1,pos2-2);
            //std::cout << mesh <<std::endl;
            result.insert({stoi(label),mesh});
        }
        file.close();
    }

    return result;
}

void convert_voxel_to_mesh(string workdir, NiftiImage& image,pair<int,string> label, TransformMatrix &from_mni_to_label){

    Surface surface = Surface();
    surface.read_volume(label.second);

    TransformMatrix from_label_to_mni = from_mni_to_label.get_inverse();


    //shrink volume
    VolumeInt* lab1_vol = (VolumeInt*)image.returnImage();

    auto label_mask = lab1_vol->label_to_mask(label.first);
    //calc centre of label
    Point centr_of_label = label_mask.center_of_mass();
    //point of center in world coordinate
    auto swap_centre = image.get_voxel_to_world().apply_transform(centr_of_label.getX(),centr_of_label.getY(),centr_of_label.getZ());

    auto  back_centre = image.get_world_to_voxel().apply_transform(swap_centre);

    auto lab_centre_mni = from_label_to_mni.apply_transform(swap_centre); //map labels centre to mni space

    //GENERATE SPHERE IN MNI COORDS!!!

    tuple<double,double,double> ride = {lab_centre_mni[0],lab_centre_mni[1],lab_centre_mni[2]};
    //tuple<double,double,double> ride = {centr_of_label.getPt()[0],centr_of_label.getPt()[1],centr_of_label.getPt()[2]};
    auto sphr = Surface::generate_sphere(50,ride,1);

    sphr.write_obj(workdir + "/"+ std::to_string(label.first) + "_cent_sphere.obj" );

    sphr.apply_transformation(from_mni_to_label);
    //sphr.write_obj(workdir + "/" + std::to_string(label.first) + "_cent_sphere.obj");
    auto W_V_trans= image.get_world_to_voxel();
    sphr.apply_transformation(W_V_trans);
    //sphr.expand_volume(30);
    //calc c
    auto mask = label_mask.int_to_double();
    //apply transformation
    //sphr.apply_transformation(from_mni_to_label);
    surface.apply_transformation(from_mni_to_label);
    //surface.write_obj(workdir + "/" + std::to_string(label.first) + "_mapped_mesh.obj");


    //sphr.write_obj(workdir + "/" + std::to_string(label.first) + "_cent_sphere.obj");
    //SHRINK SPHERE
    sphr.shrink_sphere(mask,centr_of_label.to_tuple(), 0.3);
//    sphr.smoothMesh();
    for (int i = 0;i < 10;i++) {
        //sphr.triangle_normalisation(1, 0.1);
        sphr.smoothMesh(5);
        sphr.lab_move_points(mask, 0.3,0.1);
    }
    sphr.smoothMesh(5);
    auto trans = image.get_voxel_to_world();
    sphr.apply_transformation(trans);
    sphr.apply_transformation(from_label_to_mni);


    //sphr.saveImage(workdir + "/"+ std::to_string(label.first) + "_modified_sphere.png" );
    std::cout << "Calculated volume of " << label.first << "  is " << sphr.calculate_volume() << std::endl;
    sphr.write_obj(workdir + "/" + std::to_string(label.first) + "_mni1.obj");


}

void convert_mesh_to_labels(unordered_map<int,string> meshes,string workdir,NiftiImage image){
    auto flirt_transform = TransformMatrix::read_matrix(workdir + "/combined_affine_reverse.mat");

    NiftiImage mni = NiftiImage();
    mni.read_nifti_image(workdir + "/t1_brain_to_mni_stage2_apply.nii.gz");

    NiftiImage* native = new NiftiImage();


    native->read_nifti_image(workdir + "/t1_acpc_extracted.nii.gz");


    //get flirt transform from flirt to native
    auto mni2native = TransformMatrix::convert_flirt_W_W(flirt_transform,mni,*native);
    //mni2native.getMatrix().print("res ");
    delete native;
    //read surface
    for (auto const& mesh: meshes){
        convert_voxel_to_mesh(workdir,image,mesh,mni2native);
    }


}



void test_interpolation(){
//    std::string im_n = "/mnt/c/Users/ivarh/Pictures/Untitled.bmp";
//    auto im_Read = vtkSmartPointer<vtkBMPReader>::New();
//    im_Read->SetFileName(im_n.c_str());
//    im_Read->Update();
//    auto readeddata = im_Read->GetOutput();
//    int * coord = new int(3);
//    coord[0] = 118;
//    coord[1] = 184;
//    coord[2] = 0;
//
//    ///temp line
//    auto ss = readeddata->GetDimensions();
//    auto ss2 = readeddata->GetCenter();
//    im_Read->Print(cout);
//    int *val  = static_cast<int *>(readeddata->GetScalarPointer(coord));
//    int value  = (int) (*val);
//    double * coord2 = new double (3);
//    coord2[0] = 134;
//    coord2[1] = 161;
//    coord2[2] = 0;
//    auto coeff = vtkSmartPointer<vtkImageBSplineCoefficients>::New();
//    coeff->SetInputData(readeddata);
//    coeff->Update();
//    auto check = coeff->CheckBounds(coord2);
//    auto res = coeff->Evaluate(118,184,0);
//    auto interp = vtkSmartPointer<vtkImageBSplineInterpolator>::New();
//    interp->SetSplineDegree(3);
//    interp->Initialize(coeff->GetOutput());
//    interp->Update();
//    auto res1 = interp->Interpolate(118,184,0,0);
//    double* arl;
//    interp->InterpolateIJK(coord2,arl);
//    auto interp2 = vtkSmartPointer<vtkImageBSplineInterpolator>::New();
//    interp2->Initialize(readeddata);
//    interp2->Update();
//    auto res2 = interp->Interpolate(118,184,0,0);
}

//todo remove after
void test_image( ){
    std::string label_file= "/home/varga/processing_data/Homolka/processing1/workdir/KOUT_JIRI_9675633_out/labels_t2_se_TRA_resampled.nii.gz";



    auto ni = NiftiImage();
    ni.read_nifti_image(label_file);

    VolumeInt* image = (VolumeInt*) ni.returnImage();

    auto stn = image->label_to_mask(3).get_mask_as_vector();

    auto to_mni = std::vector<std::vector<double>>();
    to_mni.emplace_back();
    to_mni.emplace_back();
    to_mni.emplace_back();
    to_mni.emplace_back();
    to_mni[0].push_back(0.98541);
    to_mni[0].push_back(-0.01442);
    to_mni[0].push_back(-0.08760);
    to_mni[0].push_back(-118.90224);

    to_mni[1].push_back(-0.03361);
    to_mni[1].push_back(1.09207);
    to_mni[1].push_back(-0.11030);
    to_mni[1].push_back(-85.92115);

    to_mni[2].push_back(0.08250
    );
    to_mni[2].push_back(-0.02879
    );
    to_mni[2].push_back(1.65038
    );
    to_mni[2].push_back(-69.56446
    );

    to_mni[3].push_back(0.00000
    );
    to_mni[3].push_back(0.00000
    );
    to_mni[3].push_back(0.00000
    );
    to_mni[3].push_back(1.00000
    );


    auto res = pySurface::calculate_label( stn,
                                           to_mni,
                                           10,1,
                                           0.3,5
                                           ,5);

    res.saveObj("/home/varga/processing_data/kj_stn.obj");
    exit(1);

}




int main(int argc, char *argv[]) {
    test_image();
    std::cout << "Hello, World!" << std::endl;

    test_interpolation();
    CLIParser parser = CLIParser();
    parser.parse_options(argc,argv);

//    Surface surface = Surface();
//    surface.read_volume("/mnt/f/fsl/src/mist-clean/data/meshes/left_red_nucleus.mim");
//    auto a = surface.getTrianglesAsVec();

    vector<string> run_param = parser.getValue("run");

    if (run_param[0]=="mesh_shrinkage"){
        vector<string> input_im = parser.getValue("i");
        NiftiImage image = NiftiImage();
        image.read_nifti_image(input_im[0]);
        input_im.clear();
        input_im = parser.getValue("labeldesk");
        auto x = image.returnImage();
        auto wd = parser.getValue("workdir");

        auto labels_mesh = parse_label_descriptor(input_im[0]);

        //read flirt transform

        convert_mesh_to_labels(labels_mesh,wd[0],image);



        cout << 101110;





        return 0;
    }

    return 0;
}
