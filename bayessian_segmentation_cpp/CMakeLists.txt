 cmake_minimum_required(VERSION 2.8)
project(bayessian_segmentation_cpp)

set(CMAKE_CXX_STANDARD 14)
 set(VTK_DIR "/home/igorv/vtk/VTK-9.0.1/build" CACHE PATH "VTK directory override" FORCE )
 find_package(VTK COMPONENTS
         vtkCommonCore
         vtkFiltersGeneral
         vtkInteractionStyle
         vtkIOLegacy
         vtkIOGeometry
         vtkIOPLY
         vtkIOXML
         vtkRenderingCore
         vtkRenderingFreeType
         vtkFiltersGeometry

         QUIET)

 if (NOT VTK_FOUND)
 message("Skipping ReadTextFile: ${VTK_NOT_FOUND_MESSAGE}")
 return ()
 endif()

add_executable(bayessian_segmentation_cpp main.cpp Surface.cpp Surface.h)
 target_link_libraries(bayessian_segmentation_cpp PRIVATE ${VTK_LIBRARIES})
 # vtk_module_autoinit is needed
 vtk_module_autoinit(
         TARGETS bayessian_segmentation_cpp
         MODULES ${VTK_LIBRARIES}
 )