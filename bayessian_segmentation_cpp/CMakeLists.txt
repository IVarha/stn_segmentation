cmake_minimum_required(VERSION 3.10)
project(bayessian_segmentation_cpp)
 set(EXEC_NAME bayessian_segmentation_cpp)
 list(APPEND CMAKE_PREFIX_PATH "/home/varga/libs/bay_segm/spdlog/build2/lib/cmake/spdlog")
set(CMAKE_CXX_STANDARD 17)
add_compile_options("-fPIC")
 set(VTK_DIR "/home/varga/VTK-9.0.1/build" CACHE PATH "VTK directory override" FORCE )
 find_package(spdlog REQUIRED)
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

         #vtkRenderingContextOpenGL2
         vtkRenderingCore
         vtkRenderingFreeType
         vtkRenderingFreeType
         vtkRenderingGL2PSOpenGL2
         vtkRenderingOpenGL2
         vtkRenderingVolumeOpenGL2

         QUIET)
 find_package(ZLIB REQUIRED)
 find_package(Armadillo REQUIRED)


 add_subdirectory(external/pybind11-2.6)
include_directories(pybind11_INCLUDE_DIRS)

 if (NOT VTK_FOUND)
 message("Skipping ReadTextFile: ${VTK_NOT_FOUND_MESSAGE}")
 return ()
 endif()
 include_directories(utils)
 include_directories(segmentation)
 include_directories(py_wrappers)
 include_directories(external/spdlog-1.8.2/include)
 include_directories(${ARMADILLO_INCLUDE_DIRS})
 add_library(Segmentation
         segmentation/Surface.cpp segmentation/Surface.h
         segmentation/NiftiImage.cpp segmentation/NiftiImage.h segmentation/Point.cpp segmentation/Point.h
         segmentation/triangleintersects.h py_wrappers/pySurface.h py_wrappers/pyNiftiImage.h)


 add_library( Utils
         utils/CLIParser.cpp utils/CLIParser.h)
 set(SOURCES utils/CLIParser.cpp segmentation/Surface.cpp
         segmentation/NiftiImage.cpp segmentation/NiftiImage.h segmentation/Point.cpp py_wrappers/pySurface.cpp py_wrappers/pyNiftiImage.cpp)

add_executable(bayessian_segmentation_cpp main.cpp ${SOURCES})
 pybind11_add_module(ExtPy pybind_test.cpp py_wrappers/pySurface.cpp )



 target_link_libraries(bayessian_segmentation_cpp PRIVATE ${VTK_LIBRARIES} ZLIB::ZLIB  ${ARMADILLO_LIBRARIES} spdlog::spdlog )
 target_link_libraries(Segmentation PUBLIC ${VTK_LIBRARIES} ZLIB::ZLIB  ${ARMADILLO_LIBRARIES} spdlog::spdlog)
 target_link_libraries(ExtPy PUBLIC ${VTK_LIBRARIES} ZLIB::ZLIB  ${ARMADILLO_LIBRARIES} Segmentation spdlog::spdlog)
 # vtk_module_autoinit is needed
 vtk_module_autoinit(
         TARGETS bayessian_segmentation_cpp
         MODULES ${VTK_LIBRARIES}
 )
