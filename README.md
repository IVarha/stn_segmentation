"# stn_segmentation" 
158-subject problematic


first step to use install VTK-9.0.1
download
1 untar
2 cd vtk lib
3 cmake ..
4 make -j<num of cores>

Make spdlog from external folder and install it :
1. сreate build folder
2. make spldog by cmake cmake -D CMAKE_INSTALL_PREFIX:PATH=<install dir> <source_dir>
3. set spdlog_DIR in bashrc

make c++ code


cmake --build /tmp/pycharm_project_545/bayessian_segmentation_cpp/build  --target all -- -j 6

Add 
add_compile_options("-fPIC")
to spdlog

genereate_meshes_from_atlas - generate mesh from atlas

