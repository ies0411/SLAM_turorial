# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.16

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/eunsoo/Downloads/study_알짜/slambook/practice2

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/eunsoo/Downloads/study_알짜/slambook/practice2/build

# Include any dependencies generated for this target.
include CMakeFiles/testimage.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/testimage.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/testimage.dir/flags.make

CMakeFiles/testimage.dir/testimage.cpp.o: CMakeFiles/testimage.dir/flags.make
CMakeFiles/testimage.dir/testimage.cpp.o: ../testimage.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/eunsoo/Downloads/study_알짜/slambook/practice2/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/testimage.dir/testimage.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/testimage.dir/testimage.cpp.o -c /home/eunsoo/Downloads/study_알짜/slambook/practice2/testimage.cpp

CMakeFiles/testimage.dir/testimage.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/testimage.dir/testimage.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/eunsoo/Downloads/study_알짜/slambook/practice2/testimage.cpp > CMakeFiles/testimage.dir/testimage.cpp.i

CMakeFiles/testimage.dir/testimage.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/testimage.dir/testimage.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/eunsoo/Downloads/study_알짜/slambook/practice2/testimage.cpp -o CMakeFiles/testimage.dir/testimage.cpp.s

# Object files for target testimage
testimage_OBJECTS = \
"CMakeFiles/testimage.dir/testimage.cpp.o"

# External object files for target testimage
testimage_EXTERNAL_OBJECTS =

testimage: CMakeFiles/testimage.dir/testimage.cpp.o
testimage: CMakeFiles/testimage.dir/build.make
testimage: /usr/local/lib/libopencv_dnn.so.3.4.5
testimage: /usr/local/lib/libopencv_ml.so.3.4.5
testimage: /usr/local/lib/libopencv_objdetect.so.3.4.5
testimage: /usr/local/lib/libopencv_shape.so.3.4.5
testimage: /usr/local/lib/libopencv_stitching.so.3.4.5
testimage: /usr/local/lib/libopencv_superres.so.3.4.5
testimage: /usr/local/lib/libopencv_videostab.so.3.4.5
testimage: /usr/lib/x86_64-linux-gnu/libfmt.a
testimage: /usr/lib/x86_64-linux-gnu/libpcl_io.so
testimage: /usr/lib/x86_64-linux-gnu/libboost_system.so
testimage: /usr/lib/x86_64-linux-gnu/libboost_filesystem.so
testimage: /usr/lib/x86_64-linux-gnu/libboost_date_time.so
testimage: /usr/lib/x86_64-linux-gnu/libboost_iostreams.so
testimage: /usr/lib/x86_64-linux-gnu/libboost_regex.so
testimage: /usr/lib/x86_64-linux-gnu/libvtkChartsCore-7.1.so.7.1p.1
testimage: /usr/lib/x86_64-linux-gnu/libvtkInfovisCore-7.1.so.7.1p.1
testimage: /usr/lib/x86_64-linux-gnu/libfreetype.so
testimage: /usr/lib/x86_64-linux-gnu/libz.so
testimage: /usr/lib/x86_64-linux-gnu/libjpeg.so
testimage: /usr/lib/x86_64-linux-gnu/libpng.so
testimage: /usr/lib/x86_64-linux-gnu/libtiff.so
testimage: /usr/lib/x86_64-linux-gnu/libexpat.so
testimage: /usr/lib/x86_64-linux-gnu/libvtkIOGeometry-7.1.so.7.1p.1
testimage: /usr/lib/x86_64-linux-gnu/libvtkIOLegacy-7.1.so.7.1p.1
testimage: /usr/lib/x86_64-linux-gnu/libvtkIOPLY-7.1.so.7.1p.1
testimage: /usr/lib/x86_64-linux-gnu/libvtkRenderingLOD-7.1.so.7.1p.1
testimage: /usr/lib/x86_64-linux-gnu/libvtkViewsContext2D-7.1.so.7.1p.1
testimage: /usr/lib/x86_64-linux-gnu/libvtkViewsCore-7.1.so.7.1p.1
testimage: /usr/lib/x86_64-linux-gnu/libvtkRenderingContextOpenGL2-7.1.so.7.1p.1
testimage: /usr/lib/x86_64-linux-gnu/libvtkRenderingOpenGL2-7.1.so.7.1p.1
testimage: /usr/local/lib/libopencv_calib3d.so.3.4.5
testimage: /usr/local/lib/libopencv_features2d.so.3.4.5
testimage: /usr/local/lib/libopencv_flann.so.3.4.5
testimage: /usr/local/lib/libopencv_highgui.so.3.4.5
testimage: /usr/local/lib/libopencv_photo.so.3.4.5
testimage: /usr/local/lib/libopencv_video.so.3.4.5
testimage: /usr/local/lib/libopencv_videoio.so.3.4.5
testimage: /usr/local/lib/libopencv_imgcodecs.so.3.4.5
testimage: /usr/local/lib/libopencv_imgproc.so.3.4.5
testimage: /usr/local/lib/libopencv_core.so.3.4.5
testimage: /usr/lib/x86_64-linux-gnu/libpcl_octree.so
testimage: /usr/lib/x86_64-linux-gnu/libpcl_common.so
testimage: /usr/lib/x86_64-linux-gnu/libvtkInteractionWidgets-7.1.so.7.1p.1
testimage: /usr/lib/x86_64-linux-gnu/libvtkFiltersModeling-7.1.so.7.1p.1
testimage: /usr/lib/x86_64-linux-gnu/libvtkInteractionStyle-7.1.so.7.1p.1
testimage: /usr/lib/x86_64-linux-gnu/libvtkFiltersExtraction-7.1.so.7.1p.1
testimage: /usr/lib/x86_64-linux-gnu/libvtkFiltersStatistics-7.1.so.7.1p.1
testimage: /usr/lib/x86_64-linux-gnu/libvtkImagingFourier-7.1.so.7.1p.1
testimage: /usr/lib/x86_64-linux-gnu/libvtkalglib-7.1.so.7.1p.1
testimage: /usr/lib/x86_64-linux-gnu/libvtkFiltersHybrid-7.1.so.7.1p.1
testimage: /usr/lib/x86_64-linux-gnu/libvtkImagingGeneral-7.1.so.7.1p.1
testimage: /usr/lib/x86_64-linux-gnu/libvtkImagingSources-7.1.so.7.1p.1
testimage: /usr/lib/x86_64-linux-gnu/libvtkImagingHybrid-7.1.so.7.1p.1
testimage: /usr/lib/x86_64-linux-gnu/libvtkRenderingAnnotation-7.1.so.7.1p.1
testimage: /usr/lib/x86_64-linux-gnu/libvtkImagingColor-7.1.so.7.1p.1
testimage: /usr/lib/x86_64-linux-gnu/libvtkRenderingVolume-7.1.so.7.1p.1
testimage: /usr/lib/x86_64-linux-gnu/libvtkIOXML-7.1.so.7.1p.1
testimage: /usr/lib/x86_64-linux-gnu/libvtkIOXMLParser-7.1.so.7.1p.1
testimage: /usr/lib/x86_64-linux-gnu/libvtkIOCore-7.1.so.7.1p.1
testimage: /usr/lib/x86_64-linux-gnu/libvtkRenderingContext2D-7.1.so.7.1p.1
testimage: /usr/lib/x86_64-linux-gnu/libvtkRenderingFreeType-7.1.so.7.1p.1
testimage: /usr/lib/x86_64-linux-gnu/libfreetype.so
testimage: /usr/lib/x86_64-linux-gnu/libvtkImagingCore-7.1.so.7.1p.1
testimage: /usr/lib/x86_64-linux-gnu/libvtkRenderingCore-7.1.so.7.1p.1
testimage: /usr/lib/x86_64-linux-gnu/libvtkCommonColor-7.1.so.7.1p.1
testimage: /usr/lib/x86_64-linux-gnu/libvtkFiltersGeometry-7.1.so.7.1p.1
testimage: /usr/lib/x86_64-linux-gnu/libvtkFiltersSources-7.1.so.7.1p.1
testimage: /usr/lib/x86_64-linux-gnu/libvtkFiltersGeneral-7.1.so.7.1p.1
testimage: /usr/lib/x86_64-linux-gnu/libvtkCommonComputationalGeometry-7.1.so.7.1p.1
testimage: /usr/lib/x86_64-linux-gnu/libvtkFiltersCore-7.1.so.7.1p.1
testimage: /usr/lib/x86_64-linux-gnu/libvtkIOImage-7.1.so.7.1p.1
testimage: /usr/lib/x86_64-linux-gnu/libvtkCommonExecutionModel-7.1.so.7.1p.1
testimage: /usr/lib/x86_64-linux-gnu/libvtkCommonDataModel-7.1.so.7.1p.1
testimage: /usr/lib/x86_64-linux-gnu/libvtkCommonTransforms-7.1.so.7.1p.1
testimage: /usr/lib/x86_64-linux-gnu/libvtkCommonMisc-7.1.so.7.1p.1
testimage: /usr/lib/x86_64-linux-gnu/libvtkCommonMath-7.1.so.7.1p.1
testimage: /usr/lib/x86_64-linux-gnu/libvtkCommonSystem-7.1.so.7.1p.1
testimage: /usr/lib/x86_64-linux-gnu/libvtkCommonCore-7.1.so.7.1p.1
testimage: /usr/lib/x86_64-linux-gnu/libvtksys-7.1.so.7.1p.1
testimage: /usr/lib/x86_64-linux-gnu/libvtkDICOMParser-7.1.so.7.1p.1
testimage: /usr/lib/x86_64-linux-gnu/libvtkmetaio-7.1.so.7.1p.1
testimage: /usr/lib/x86_64-linux-gnu/libz.so
testimage: /usr/lib/x86_64-linux-gnu/libGLEW.so
testimage: /usr/lib/x86_64-linux-gnu/libSM.so
testimage: /usr/lib/x86_64-linux-gnu/libICE.so
testimage: /usr/lib/x86_64-linux-gnu/libX11.so
testimage: /usr/lib/x86_64-linux-gnu/libXext.so
testimage: /usr/lib/x86_64-linux-gnu/libXt.so
testimage: CMakeFiles/testimage.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/eunsoo/Downloads/study_알짜/slambook/practice2/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable testimage"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/testimage.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/testimage.dir/build: testimage

.PHONY : CMakeFiles/testimage.dir/build

CMakeFiles/testimage.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/testimage.dir/cmake_clean.cmake
.PHONY : CMakeFiles/testimage.dir/clean

CMakeFiles/testimage.dir/depend:
	cd /home/eunsoo/Downloads/study_알짜/slambook/practice2/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/eunsoo/Downloads/study_알짜/slambook/practice2 /home/eunsoo/Downloads/study_알짜/slambook/practice2 /home/eunsoo/Downloads/study_알짜/slambook/practice2/build /home/eunsoo/Downloads/study_알짜/slambook/practice2/build /home/eunsoo/Downloads/study_알짜/slambook/practice2/build/CMakeFiles/testimage.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/testimage.dir/depend

