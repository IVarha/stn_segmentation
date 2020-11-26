 cmake_minimum_required(VERSION 2.8)
project(bayessian_segmentation_cpp)

set(CMAKE_CXX_STANDARD 14)
 set(VTK_DIR "/home/igorv/vtk/VTK-9.0.1/build" CACHE PATH "VTK directory override" FORCE )
 find_package(VTK COMPONENTS
         vtkCommonCore
         vtkCommonDataModel
         vtkFiltersGeneral
         vtkFiltersGeometry
         vtkFiltersSources
         vtkFiltersModeling
         vtkInteractionStyle
         vtkInteractionImage
         vtkIOLegacy
         vtkIOGeometry
         vtkIOPLY
         vtkIOXML
         vtkIOImage
         vtkRenderingCore
         vtkRenderingFreeType


         QUIET)
 find_package(ZLIB REQUIRED)
 find_package(Armadillo REQUIRED)


 if (NOT VTK_FOUND)
 message("Skipping ReadTextFile: ${VTK_NOT_FOUND_MESSAGE}")
 return ()
 endif()
 find_package(Armadillo REQUIRED)
 include_directories(utils)
 include_directories(segmentation)
 include_directories(${ARMADILLO_INCLUDE_DIRS})
 add_library(Segmentation
         Surface.cpp Surface.h
         segmentation/NiftiImage.cpp segmentation/NiftiImage.h segmentation/Point.cpp segmentation/Point.h)


 add_library( Utils
         utils/CLIParser.cpp utils/CLIParser.h)
 set(SOURCES utils/CLIParser.cpp Surface.cpp
         segmentation/NiftiImage.cpp segmentation/NiftiImage.h segmentation/Point.cpp )

add_executable(bayessian_segmentation_cpp main.cpp ${SOURCES})
 target_link_libraries(bayessian_segmentation_cpp PRIVATE ${VTK_LIBRARIES} ZLIB::ZLIB  ${ARMADILLO_LIBRARIES})
 target_link_libraries(Segmentation PRIVATE ${VTK_LIBRARIES}  ${ARMADILLO_LIBRARIES})
 # vtk_module_autoinit is needed
 vtk_module_autoinit(
         TARGETS bayessian_segmentation_cpp
         MODULES ${VTK_LIBRARIES}
 )
