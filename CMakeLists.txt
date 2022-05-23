CMAKE_MINIMUM_REQUIRED (VERSION 3.15 )
project(02)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++17")
SET(G2O_LIBS g2o_cli g2o_ext_freeglut_minimal g2o_simulator g2o_solver_slam2d_linear g2o_types_icp g2o_types_slam2d g2o_core g2o_interface g2o_solver_csparse g2o_solver_structure_only g2o_types_sba g2o_types_slam3d g2o_csparse_extension g2o_opengl_helper g2o_solver_dense g2o_stuff g2o_types_sclam2d g2o_parser g2o_solver_pcg g2o_types_data g2o_types_sim3 cxsparse )

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)

find_package(FMT REQUIRED)
# find_package( Pangolin )
find_package( Sophus)
find_package( OpenCV 3 REQUIRED)
find_package( PCL REQUIRED COMPONENT common io )
find_package( Ceres REQUIRED)
find_package( G2O REQUIRED)
find_package(CSparse REQUIRED)

include_directories(
  ${PCL_INCLUDE_DIRS}
  ${CERES_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${Sophus_INCLUDE_DIRS}
  ${G2O_INCLUDE_DIRS}
  ${CSPARSE_INCLUDE_DIR}
  "/usr/include/eigen3/"
  )


# include_directories()
# include_directories( "/usr/local/include/sophus" )
# include_directories( ${Pangolin_INCLUDE_DIRS} )


add_definitions( ${PCL_DEFINITIONS} )

# add_library(libhelloSLAM libhelloSLAM.cpp)

add_executable(testEigen testEigen.cpp)
add_executable(ceresTutorial ceresTutorial.cpp)
add_executable(testimage testimage.cpp)
add_executable(featureTutorial featureTutorial.cpp)
add_executable(testSophus testSophus.cpp)
add_executable(testg2o testg2o.cpp)
# add_executable(joinMap joinMap.cpp)
add_executable(testceres testceres.cpp)

target_link_libraries(testSophus ${Sophus_LIBRARIES} fmt::fmt )
target_link_libraries(featureTutorial ${OpenCV_LIBS} ${PCL_LIBRARIES} ${CSPARSE_LIBRARY}
g2o_core g2o_stuff g2o_types_sba g2o_csparse_extension ${G2O_LIBS})
target_link_libraries(testg2o ${OpenCV_LIBS} g2o_core g2o_stuff g2o_types_sba g2o_csparse_extension ${G2O_LIBS})
target_link_libraries(testimage ${OpenCV_LIBS} fmt::fmt ${PCL_LIBRARIES} )
# target_link_libraries(sophusSLAM ${Sophus_LIBRARIES} fmt::fmt)
target_link_libraries(testceres ${CERES_LIBRARIES} ${OpenCV_LIBS} )
target_link_libraries(ceresTutorial ${CERES_LIBRARIES} ${OpenCV_LIBS} )

# target_link_libraries(eigenSLAM ${Pangolin_LIBRARIES} )



