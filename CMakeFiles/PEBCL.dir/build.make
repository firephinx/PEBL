# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 2.8

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
CMAKE_SOURCE_DIR = /home/klz1/Documents/PEBCL

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/klz1/Documents/PEBCL

# Include any dependencies generated for this target.
include CMakeFiles/PEBCL.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/PEBCL.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/PEBCL.dir/flags.make

CMakeFiles/PEBCL.dir/PEBCL.cpp.o: CMakeFiles/PEBCL.dir/flags.make
CMakeFiles/PEBCL.dir/PEBCL.cpp.o: PEBCL.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/klz1/Documents/PEBCL/CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/PEBCL.dir/PEBCL.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/PEBCL.dir/PEBCL.cpp.o -c /home/klz1/Documents/PEBCL/PEBCL.cpp

CMakeFiles/PEBCL.dir/PEBCL.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/PEBCL.dir/PEBCL.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/klz1/Documents/PEBCL/PEBCL.cpp > CMakeFiles/PEBCL.dir/PEBCL.cpp.i

CMakeFiles/PEBCL.dir/PEBCL.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/PEBCL.dir/PEBCL.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/klz1/Documents/PEBCL/PEBCL.cpp -o CMakeFiles/PEBCL.dir/PEBCL.cpp.s

CMakeFiles/PEBCL.dir/PEBCL.cpp.o.requires:
.PHONY : CMakeFiles/PEBCL.dir/PEBCL.cpp.o.requires

CMakeFiles/PEBCL.dir/PEBCL.cpp.o.provides: CMakeFiles/PEBCL.dir/PEBCL.cpp.o.requires
	$(MAKE) -f CMakeFiles/PEBCL.dir/build.make CMakeFiles/PEBCL.dir/PEBCL.cpp.o.provides.build
.PHONY : CMakeFiles/PEBCL.dir/PEBCL.cpp.o.provides

CMakeFiles/PEBCL.dir/PEBCL.cpp.o.provides.build: CMakeFiles/PEBCL.dir/PEBCL.cpp.o

# Object files for target PEBCL
PEBCL_OBJECTS = \
"CMakeFiles/PEBCL.dir/PEBCL.cpp.o"

# External object files for target PEBCL
PEBCL_EXTERNAL_OBJECTS =

PEBCL: CMakeFiles/PEBCL.dir/PEBCL.cpp.o
PEBCL: CMakeFiles/PEBCL.dir/build.make
PEBCL: /usr/lib/x86_64-linux-gnu/libboost_system.so
PEBCL: /usr/lib/x86_64-linux-gnu/libboost_filesystem.so
PEBCL: /usr/lib/x86_64-linux-gnu/libboost_thread.so
PEBCL: /usr/lib/x86_64-linux-gnu/libboost_date_time.so
PEBCL: /usr/lib/x86_64-linux-gnu/libboost_iostreams.so
PEBCL: /usr/lib/x86_64-linux-gnu/libboost_serialization.so
PEBCL: /usr/lib/x86_64-linux-gnu/libpthread.so
PEBCL: /usr/lib/libpcl_common.so
PEBCL: /usr/lib/x86_64-linux-gnu/libflann_cpp_s.a
PEBCL: /usr/lib/libpcl_kdtree.so
PEBCL: /usr/lib/libpcl_octree.so
PEBCL: /usr/lib/libpcl_search.so
PEBCL: /usr/lib/x86_64-linux-gnu/libqhull.so
PEBCL: /usr/lib/libpcl_surface.so
PEBCL: /usr/lib/libpcl_sample_consensus.so
PEBCL: /usr/lib/libpcl_filters.so
PEBCL: /usr/lib/libpcl_features.so
PEBCL: /usr/lib/libpcl_segmentation.so
PEBCL: /usr/lib/libOpenNI.so
PEBCL: /usr/lib/libvtkCommon.so.5.8.0
PEBCL: /usr/lib/libvtkRendering.so.5.8.0
PEBCL: /usr/lib/libvtkHybrid.so.5.8.0
PEBCL: /usr/lib/libvtkCharts.so.5.8.0
PEBCL: /usr/lib/libpcl_io.so
PEBCL: /usr/lib/libpcl_registration.so
PEBCL: /usr/lib/libpcl_keypoints.so
PEBCL: /usr/lib/libpcl_recognition.so
PEBCL: /usr/lib/libpcl_visualization.so
PEBCL: /usr/lib/libpcl_people.so
PEBCL: /usr/lib/libpcl_outofcore.so
PEBCL: /usr/lib/libpcl_tracking.so
PEBCL: /usr/lib/libpcl_apps.so
PEBCL: /usr/lib/x86_64-linux-gnu/libboost_system.so
PEBCL: /usr/lib/x86_64-linux-gnu/libboost_filesystem.so
PEBCL: /usr/lib/x86_64-linux-gnu/libboost_thread.so
PEBCL: /usr/lib/x86_64-linux-gnu/libboost_date_time.so
PEBCL: /usr/lib/x86_64-linux-gnu/libboost_iostreams.so
PEBCL: /usr/lib/x86_64-linux-gnu/libboost_serialization.so
PEBCL: /usr/lib/x86_64-linux-gnu/libpthread.so
PEBCL: /usr/lib/x86_64-linux-gnu/libqhull.so
PEBCL: /usr/lib/libOpenNI.so
PEBCL: /usr/lib/x86_64-linux-gnu/libflann_cpp_s.a
PEBCL: /usr/lib/libvtkCommon.so.5.8.0
PEBCL: /usr/lib/libvtkRendering.so.5.8.0
PEBCL: /usr/lib/libvtkHybrid.so.5.8.0
PEBCL: /usr/lib/libvtkCharts.so.5.8.0
PEBCL: /usr/lib/libpcl_common.so
PEBCL: /usr/lib/libpcl_kdtree.so
PEBCL: /usr/lib/libpcl_octree.so
PEBCL: /usr/lib/libpcl_search.so
PEBCL: /usr/lib/libpcl_surface.so
PEBCL: /usr/lib/libpcl_sample_consensus.so
PEBCL: /usr/lib/libpcl_filters.so
PEBCL: /usr/lib/libpcl_features.so
PEBCL: /usr/lib/libpcl_segmentation.so
PEBCL: /usr/lib/libpcl_io.so
PEBCL: /usr/lib/libpcl_registration.so
PEBCL: /usr/lib/libpcl_keypoints.so
PEBCL: /usr/lib/libpcl_recognition.so
PEBCL: /usr/lib/libpcl_visualization.so
PEBCL: /usr/lib/libpcl_people.so
PEBCL: /usr/lib/libpcl_outofcore.so
PEBCL: /usr/lib/libpcl_tracking.so
PEBCL: /usr/lib/libpcl_apps.so
PEBCL: /usr/lib/libvtkViews.so.5.8.0
PEBCL: /usr/lib/libvtkInfovis.so.5.8.0
PEBCL: /usr/lib/libvtkWidgets.so.5.8.0
PEBCL: /usr/lib/libvtkHybrid.so.5.8.0
PEBCL: /usr/lib/libvtkParallel.so.5.8.0
PEBCL: /usr/lib/libvtkVolumeRendering.so.5.8.0
PEBCL: /usr/lib/libvtkRendering.so.5.8.0
PEBCL: /usr/lib/libvtkGraphics.so.5.8.0
PEBCL: /usr/lib/libvtkImaging.so.5.8.0
PEBCL: /usr/lib/libvtkIO.so.5.8.0
PEBCL: /usr/lib/libvtkFiltering.so.5.8.0
PEBCL: /usr/lib/libvtkCommon.so.5.8.0
PEBCL: /usr/lib/libvtksys.so.5.8.0
PEBCL: CMakeFiles/PEBCL.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --red --bold "Linking CXX executable PEBCL"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/PEBCL.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/PEBCL.dir/build: PEBCL
.PHONY : CMakeFiles/PEBCL.dir/build

CMakeFiles/PEBCL.dir/requires: CMakeFiles/PEBCL.dir/PEBCL.cpp.o.requires
.PHONY : CMakeFiles/PEBCL.dir/requires

CMakeFiles/PEBCL.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/PEBCL.dir/cmake_clean.cmake
.PHONY : CMakeFiles/PEBCL.dir/clean

CMakeFiles/PEBCL.dir/depend:
	cd /home/klz1/Documents/PEBCL && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/klz1/Documents/PEBCL /home/klz1/Documents/PEBCL /home/klz1/Documents/PEBCL /home/klz1/Documents/PEBCL /home/klz1/Documents/PEBCL/CMakeFiles/PEBCL.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/PEBCL.dir/depend

