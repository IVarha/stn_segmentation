#include <iostream>
#include "Surface.h"
#include "CLIParser.h"
#include "NiftiImage.h"

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
            auto mesh = line.substr(pos+1,line.size()-3);
            result.insert({stoi(label),mesh});
        }
        file.close();
    }

    return result;
}

void convert_voxel_to_mesh(string workdir, NiftiImage image,pair<int,string> label, TransformMatrix &from_mni_to_label){

    Surface surface = Surface();
    surface.read_volume(label.second);




    //shrink volume
    VolumeInt* lab1_vol = (VolumeInt*)image.returnImage();

    auto label_mask = lab1_vol->label_to_mask(label.first);
    //calc centre of label
    Point centr_of_mesh = label_mask.center_of_mass();
    auto swap_centre = image.get_voxel_to_world().apply_transform(centr_of_mesh.getPt()[0],centr_of_mesh.getPt()[1],centr_of_mesh.getPt()[2]);


    //GENERATE SPHERE IN VOXEL COORDS!!!

    //tuple<double,double,double> ride = {swap_centre[0],swap_centre[1],swap_centre[2]};
    tuple<double,double,double> ride = {centr_of_mesh.getPt()[0],centr_of_mesh.getPt()[1],centr_of_mesh.getPt()[2]};
    auto sphr = Surface::generate_sphere(10,ride);
    //sphr.expand_volume(30);
    //calc c
    auto mask = label_mask.int_to_double();
    //apply transformation
    //sphr.apply_transformation(from_mni_to_label);
    surface.apply_transformation(from_mni_to_label);
    surface.write_obj(workdir + "/" + std::to_string(label.first) + "_mapped_mesh.obj");

    sphr.write_obj(workdir + "/" + std::to_string(label.first) + "_cent_sphere.obj");
   sphr.shrink_sphere(mask,ride, 0.5);

    auto trans = image.get_voxel_to_world();
    sphr.apply_transformation(trans);
    sphr.write_obj(workdir + "/" + std::to_string(label.first) + "_1.obj");


}

void convert_mesh_to_labels(unordered_map<int,string> meshes,string workdir,NiftiImage image){
    auto flirt_transform = TransformMatrix::read_matrix(workdir + "/combined_affine_reverse.mat");

    NiftiImage mni = NiftiImage();
    mni.read_nifti_image(workdir + "/t1_brain_to_mni_stage2_apply.nii.gz");
    NiftiImage native = NiftiImage();
    native.read_nifti_image(workdir + "/t1_acpc_extracted.nii.gz");


    //get flirt transform from flirt to native
    auto mni2native = TransformMatrix::convert_flirt_W_W(flirt_transform,mni,native);
    //mni2native.getMatrix().print("res ");

    //read surface
    for (auto const& mesh: meshes){
        convert_voxel_to_mesh(workdir,image,mesh,mni2native);
    }




    auto res = mni2native.vox_to_mm(75,77,157);
    cout << get<0>(res) << "  " << get<1>(res) << " " << get<2> (res) ;

    int k = 1;

}

int main(int argc, char *argv[]) {
    std::cout << "Hello, World!" << std::endl;


    CLIParser parser = CLIParser();
    parser.parse_options(argc,argv);


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
    Surface surface = Surface();
    surface.read_volume("/mnt/f/fsl/src/mist-clean/data/meshes/left_red_nucleus.mim");
    return 0;
}
