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
include test/googletest/googlemock/CMakeFiles/gmock_main.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include test/googletest/googlemock/CMakeFiles/gmock_main.dir/compiler_depend.make

# Include the progress variables for this target.
include test/googletest/googlemock/CMakeFiles/gmock_main.dir/progress.make

# Include the compile flags for this target's objects.
include test/googletest/googlemock/CMakeFiles/gmock_main.dir/flags.make

test/googletest/googlemock/CMakeFiles/gmock_main.dir/src/gmock_main.cc.o: test/googletest/googlemock/CMakeFiles/gmock_main.dir/flags.make
test/googletest/googlemock/CMakeFiles/gmock_main.dir/src/gmock_main.cc.o: ../test/googletest/googlemock/src/gmock_main.cc
test/googletest/googlemock/CMakeFiles/gmock_main.dir/src/gmock_main.cc.o: test/googletest/googlemock/CMakeFiles/gmock_main.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/hpj123/project/project_visnav/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object test/googletest/googlemock/CMakeFiles/gmock_main.dir/src/gmock_main.cc.o"
	cd /Users/hpj123/project/project_visnav/build/test/googletest/googlemock && /usr/local/bin/ccache /usr/bin/clang++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT test/googletest/googlemock/CMakeFiles/gmock_main.dir/src/gmock_main.cc.o -MF CMakeFiles/gmock_main.dir/src/gmock_main.cc.o.d -o CMakeFiles/gmock_main.dir/src/gmock_main.cc.o -c /Users/hpj123/project/project_visnav/test/googletest/googlemock/src/gmock_main.cc

test/googletest/googlemock/CMakeFiles/gmock_main.dir/src/gmock_main.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/gmock_main.dir/src/gmock_main.cc.i"
	cd /Users/hpj123/project/project_visnav/build/test/googletest/googlemock && /usr/bin/clang++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/hpj123/project/project_visnav/test/googletest/googlemock/src/gmock_main.cc > CMakeFiles/gmock_main.dir/src/gmock_main.cc.i

test/googletest/googlemock/CMakeFiles/gmock_main.dir/src/gmock_main.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/gmock_main.dir/src/gmock_main.cc.s"
	cd /Users/hpj123/project/project_visnav/build/test/googletest/googlemock && /usr/bin/clang++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/hpj123/project/project_visnav/test/googletest/googlemock/src/gmock_main.cc -o CMakeFiles/gmock_main.dir/src/gmock_main.cc.s

# Object files for target gmock_main
gmock_main_OBJECTS = \
"CMakeFiles/gmock_main.dir/src/gmock_main.cc.o"

# External object files for target gmock_main
gmock_main_EXTERNAL_OBJECTS =

lib/libgmock_main.a: test/googletest/googlemock/CMakeFiles/gmock_main.dir/src/gmock_main.cc.o
lib/libgmock_main.a: test/googletest/googlemock/CMakeFiles/gmock_main.dir/build.make
lib/libgmock_main.a: test/googletest/googlemock/CMakeFiles/gmock_main.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/Users/hpj123/project/project_visnav/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX static library ../../../lib/libgmock_main.a"
	cd /Users/hpj123/project/project_visnav/build/test/googletest/googlemock && $(CMAKE_COMMAND) -P CMakeFiles/gmock_main.dir/cmake_clean_target.cmake
	cd /Users/hpj123/project/project_visnav/build/test/googletest/googlemock && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/gmock_main.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
test/googletest/googlemock/CMakeFiles/gmock_main.dir/build: lib/libgmock_main.a
.PHONY : test/googletest/googlemock/CMakeFiles/gmock_main.dir/build

test/googletest/googlemock/CMakeFiles/gmock_main.dir/clean:
	cd /Users/hpj123/project/project_visnav/build/test/googletest/googlemock && $(CMAKE_COMMAND) -P CMakeFiles/gmock_main.dir/cmake_clean.cmake
.PHONY : test/googletest/googlemock/CMakeFiles/gmock_main.dir/clean

test/googletest/googlemock/CMakeFiles/gmock_main.dir/depend:
	cd /Users/hpj123/project/project_visnav/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/hpj123/project/project_visnav /Users/hpj123/project/project_visnav/test/googletest/googlemock /Users/hpj123/project/project_visnav/build /Users/hpj123/project/project_visnav/build/test/googletest/googlemock /Users/hpj123/project/project_visnav/build/test/googletest/googlemock/CMakeFiles/gmock_main.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : test/googletest/googlemock/CMakeFiles/gmock_main.dir/depend

