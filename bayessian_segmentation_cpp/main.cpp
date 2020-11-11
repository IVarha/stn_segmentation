#include <iostream>
#include "Surface.h"
#include "CLIParser.h"
#include "NiftiImage.h"

using namespace std;
int main(int argc, char *argv[]) {
    std::cout << "Hello, World!" << std::endl;


    CLIParser parser = CLIParser();
    parser.parse_options(argc,argv);


    vector<string> run_param = parser.getValue("run");

    if (run_param[0]=="mesh_shrinkage"){
        vector<string> input_im = parser.getValue("i");
        NiftiImage image = NiftiImage();
        image.read_nifti_image(input_im[0]);


       cout << 101110;





        return 0;
    }
    Surface surface = Surface();
    surface.read_volume("/mnt/f/fsl/src/mist-clean/data/meshes/left_red_nucleus.mim");
    return 0;
}
