"# stn_segmentation" 
158-subject problematic


first step to use install VTK-9.0.1
download
1 untar
2 cd vtk lib
3 cmake ..
4 make -j<num of cores>

Add 
add_compile_options("-fPIC")
to spdlog

genereate_meshes_from_atlas - generate mesh from atlas

