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
CMAKE_COMMAND = /opt/local/bin/cmake

# The command to remove a file.
RM = /opt/local/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /Users/hpj123/project/project_visnav/thirdparty/Pangolin

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/hpj123/project/project_visnav/thirdparty/build-Pangolin

# Include any dependencies generated for this target.
include tools/VideoConvert/CMakeFiles/VideoConvert.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include tools/VideoConvert/CMakeFiles/VideoConvert.dir/compiler_depend.make

# Include the progress variables for this target.
include tools/VideoConvert/CMakeFiles/VideoConvert.dir/progress.make

# Include the compile flags for this target's objects.
include tools/VideoConvert/CMakeFiles/VideoConvert.dir/flags.make

tools/VideoConvert/CMakeFiles/VideoConvert.dir/main.cpp.o: tools/VideoConvert/CMakeFiles/VideoConvert.dir/flags.make
tools/VideoConvert/CMakeFiles/VideoConvert.dir/main.cpp.o: /Users/hpj123/project/project_visnav/thirdparty/Pangolin/tools/VideoConvert/main.cpp
tools/VideoConvert/CMakeFiles/VideoConvert.dir/main.cpp.o: tools/VideoConvert/CMakeFiles/VideoConvert.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/hpj123/project/project_visnav/thirdparty/build-Pangolin/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object tools/VideoConvert/CMakeFiles/VideoConvert.dir/main.cpp.o"
	cd /Users/hpj123/project/project_visnav/thirdparty/build-Pangolin/tools/VideoConvert && ccache /usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT tools/VideoConvert/CMakeFiles/VideoConvert.dir/main.cpp.o -MF CMakeFiles/VideoConvert.dir/main.cpp.o.d -o CMakeFiles/VideoConvert.dir/main.cpp.o -c /Users/hpj123/project/project_visnav/thirdparty/Pangolin/tools/VideoConvert/main.cpp

tools/VideoConvert/CMakeFiles/VideoConvert.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/VideoConvert.dir/main.cpp.i"
	cd /Users/hpj123/project/project_visnav/thirdparty/build-Pangolin/tools/VideoConvert && /usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/hpj123/project/project_visnav/thirdparty/Pangolin/tools/VideoConvert/main.cpp > CMakeFiles/VideoConvert.dir/main.cpp.i

tools/VideoConvert/CMakeFiles/VideoConvert.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/VideoConvert.dir/main.cpp.s"
	cd /Users/hpj123/project/project_visnav/thirdparty/build-Pangolin/tools/VideoConvert && /usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/hpj123/project/project_visnav/thirdparty/Pangolin/tools/VideoConvert/main.cpp -o CMakeFiles/VideoConvert.dir/main.cpp.s

# Object files for target VideoConvert
VideoConvert_OBJECTS = \
"CMakeFiles/VideoConvert.dir/main.cpp.o"

# External object files for target VideoConvert
VideoConvert_EXTERNAL_OBJECTS =

tools/VideoConvert/VideoConvert: tools/VideoConvert/CMakeFiles/VideoConvert.dir/main.cpp.o
tools/VideoConvert/VideoConvert: tools/VideoConvert/CMakeFiles/VideoConvert.dir/build.make
tools/VideoConvert/VideoConvert: src/libpangolin.dylib
tools/VideoConvert/VideoConvert: /usr/local/lib/libGLEW.dylib
tools/VideoConvert/VideoConvert: /opt/local/lib/libavcodec.dylib
tools/VideoConvert/VideoConvert: /opt/local/lib/libavformat.dylib
tools/VideoConvert/VideoConvert: /opt/local/lib/libavutil.dylib
tools/VideoConvert/VideoConvert: /opt/local/lib/libswscale.dylib
tools/VideoConvert/VideoConvert: /opt/local/lib/libavdevice.dylib
tools/VideoConvert/VideoConvert: /opt/local/lib/libpng.dylib
tools/VideoConvert/VideoConvert: /opt/local/lib/libz.dylib
tools/VideoConvert/VideoConvert: /opt/local/lib/libjpeg.dylib
tools/VideoConvert/VideoConvert: /opt/local/lib/libtiff.dylib
tools/VideoConvert/VideoConvert: /opt/local/lib/libzstd.dylib
tools/VideoConvert/VideoConvert: /opt/local/lib/liblz4.dylib
tools/VideoConvert/VideoConvert: tools/VideoConvert/CMakeFiles/VideoConvert.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/Users/hpj123/project/project_visnav/thirdparty/build-Pangolin/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable VideoConvert"
	cd /Users/hpj123/project/project_visnav/thirdparty/build-Pangolin/tools/VideoConvert && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/VideoConvert.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
tools/VideoConvert/CMakeFiles/VideoConvert.dir/build: tools/VideoConvert/VideoConvert
.PHONY : tools/VideoConvert/CMakeFiles/VideoConvert.dir/build

tools/VideoConvert/CMakeFiles/VideoConvert.dir/clean:
	cd /Users/hpj123/project/project_visnav/thirdparty/build-Pangolin/tools/VideoConvert && $(CMAKE_COMMAND) -P CMakeFiles/VideoConvert.dir/cmake_clean.cmake
.PHONY : tools/VideoConvert/CMakeFiles/VideoConvert.dir/clean

tools/VideoConvert/CMakeFiles/VideoConvert.dir/depend:
	cd /Users/hpj123/project/project_visnav/thirdparty/build-Pangolin && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/hpj123/project/project_visnav/thirdparty/Pangolin /Users/hpj123/project/project_visnav/thirdparty/Pangolin/tools/VideoConvert /Users/hpj123/project/project_visnav/thirdparty/build-Pangolin /Users/hpj123/project/project_visnav/thirdparty/build-Pangolin/tools/VideoConvert /Users/hpj123/project/project_visnav/thirdparty/build-Pangolin/tools/VideoConvert/CMakeFiles/VideoConvert.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : tools/VideoConvert/CMakeFiles/VideoConvert.dir/depend

