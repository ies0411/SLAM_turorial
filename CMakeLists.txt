cmake_minimum_required( VERSION 2.8 )
project(02)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++17")


find_package(FMT REQUIRED)
# find_package( Pangolin )
find_package( Sophus)
find_package( OpenCV 3 REQUIRED)
find_package( PCL REQUIRED COMPONENT common io )
find_package( Ceres REQUIRED)
# find_package( G2O REQUIRED)

include_directories(
  ${PCL_INCLUDE_DIRS}
  ${CERES_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${Sophus_INCLUDE_DIRS}
  )

include_directories("/usr/include/eigen3/")
# include_directories( "/usr/local/include/sophus" )
# include_directories( ${Pangolin_INCLUDE_DIRS} )
# include_directories(${G2O_INCLUDE_DIRS})

add_definitions( ${PCL_DEFINITIONS} )

# add_library(libhelloSLAM libhelloSLAM.cpp)

add_executable(testEigen testEigen.cpp)
add_executable(ceresTutorial ceresTutorial.cpp)
add_executable(testimage testimage.cpp)
add_executable(featureTutorial featureTutorial.cpp)
add_executable(testSophus testSophus.cpp)
# add_executable(imageSLAM imageSLAM.cpp)
# add_executable(joinMap joinMap.cpp)
add_executable(testceres testceres.cpp)

target_link_libraries(testSophus ${Sophus_LIBRARIES} fmt::fmt )
target_link_libraries(featureTutorial ${OpenCV_LIBS} ${PCL_LIBRARIES} )
# target_link_libraries(imageSLAM ${OpenCV_LIBS})
target_link_libraries(testimage ${OpenCV_LIBS} fmt::fmt ${PCL_LIBRARIES} )
# target_link_libraries(sophusSLAM ${Sophus_LIBRARIES} fmt::fmt)
target_link_libraries(testceres ${CERES_LIBRARIES} ${OpenCV_LIBS} )
target_link_libraries(ceresTutorial ${CERES_LIBRARIES} ${OpenCV_LIBS} )

# target_link_libraries(eigenSLAM ${Pangolin_LIBRARIES} )



