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


void convert_mesh_to_labels(unordered_map<int,string> meshes){


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

        auto labels_mesh = parse_label_descriptor(input_im[0]);




        cout << 101110;





        return 0;
    }
    Surface surface = Surface();
    surface.read_volume("/mnt/f/fsl/src/mist-clean/data/meshes/left_red_nucleus.mim");
    return 0;
}
