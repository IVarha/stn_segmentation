#include <iostream>
#include "Surface.h"


int main() {
    std::cout << "Hello, World!" << std::endl;

    Surface surface = Surface();
    surface.read_volume("/mnt/f/fsl/src/mist-clean/data/meshes/left_red_nucleus.mim");
    return 0;
}
