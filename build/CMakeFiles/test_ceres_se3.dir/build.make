# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.20

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /Applications/CMake.app/Contents/bin/cmake

# The command to remove a file.
RM = /Applications/CMake.app/Contents/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /Users/hpj123/project/project_visnav

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/hpj123/project/project_visnav/build

# Include any dependencies generated for this target.
include CMakeFiles/test_ceres_se3.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/test_ceres_se3.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/test_ceres_se3.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/test_ceres_se3.dir/flags.make

CMakeFiles/test_ceres_se3.dir/src/test_ceres_se3.cpp.o: CMakeFiles/test_ceres_se3.dir/flags.make
CMakeFiles/test_ceres_se3.dir/src/test_ceres_se3.cpp.o: ../src/test_ceres_se3.cpp
CMakeFiles/test_ceres_se3.dir/src/test_ceres_se3.cpp.o: CMakeFiles/test_ceres_se3.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/hpj123/project/project_visnav/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/test_ceres_se3.dir/src/test_ceres_se3.cpp.o"
	/usr/local/bin/ccache /usr/bin/clang++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/test_ceres_se3.dir/src/test_ceres_se3.cpp.o -MF CMakeFiles/test_ceres_se3.dir/src/test_ceres_se3.cpp.o.d -o CMakeFiles/test_ceres_se3.dir/src/test_ceres_se3.cpp.o -c /Users/hpj123/project/project_visnav/src/test_ceres_se3.cpp

CMakeFiles/test_ceres_se3.dir/src/test_ceres_se3.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/test_ceres_se3.dir/src/test_ceres_se3.cpp.i"
	/usr/bin/clang++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/hpj123/project/project_visnav/src/test_ceres_se3.cpp > CMakeFiles/test_ceres_se3.dir/src/test_ceres_se3.cpp.i

CMakeFiles/test_ceres_se3.dir/src/test_ceres_se3.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/test_ceres_se3.dir/src/test_ceres_se3.cpp.s"
	/usr/bin/clang++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/hpj123/project/project_visnav/src/test_ceres_se3.cpp -o CMakeFiles/test_ceres_se3.dir/src/test_ceres_se3.cpp.s

# Object files for target test_ceres_se3
test_ceres_se3_OBJECTS = \
"CMakeFiles/test_ceres_se3.dir/src/test_ceres_se3.cpp.o"

# External object files for target test_ceres_se3
test_ceres_se3_EXTERNAL_OBJECTS =

test_ceres_se3: CMakeFiles/test_ceres_se3.dir/src/test_ceres_se3.cpp.o
test_ceres_se3: CMakeFiles/test_ceres_se3.dir/build.make
test_ceres_se3: ../thirdparty/build-ceres-solver/lib/libceres.a
test_ceres_se3: /usr/local/lib/libglog.0.5.0.dylib
test_ceres_se3: /usr/local/lib/libgflags.2.2.2.dylib
test_ceres_se3: /usr/local/lib/libspqr.dylib
test_ceres_se3: /usr/local/lib/libtbb.dylib
test_ceres_se3: /usr/local/lib/libcholmod.dylib
test_ceres_se3: /usr/local/lib/libccolamd.dylib
test_ceres_se3: /usr/local/lib/libcamd.dylib
test_ceres_se3: /usr/local/lib/libcolamd.dylib
test_ceres_se3: /usr/local/lib/libamd.dylib
test_ceres_se3: /opt/local/lib/libopenblas.dylib
test_ceres_se3: /usr/local/lib/libsuitesparseconfig.dylib
test_ceres_se3: /usr/local/lib/libmetis.dylib
test_ceres_se3: /usr/local/lib/libcxsparse.dylib
test_ceres_se3: /opt/local/lib/libopenblas.dylib
test_ceres_se3: /usr/local/lib/libsuitesparseconfig.dylib
test_ceres_se3: /usr/local/lib/libmetis.dylib
test_ceres_se3: /usr/local/lib/libcxsparse.dylib
test_ceres_se3: CMakeFiles/test_ceres_se3.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/Users/hpj123/project/project_visnav/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable test_ceres_se3"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/test_ceres_se3.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/test_ceres_se3.dir/build: test_ceres_se3
.PHONY : CMakeFiles/test_ceres_se3.dir/build

CMakeFiles/test_ceres_se3.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/test_ceres_se3.dir/cmake_clean.cmake
.PHONY : CMakeFiles/test_ceres_se3.dir/clean

CMakeFiles/test_ceres_se3.dir/depend:
	cd /Users/hpj123/project/project_visnav/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/hpj123/project/project_visnav /Users/hpj123/project/project_visnav /Users/hpj123/project/project_visnav/build /Users/hpj123/project/project_visnav/build /Users/hpj123/project/project_visnav/build/CMakeFiles/test_ceres_se3.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/test_ceres_se3.dir/depend
