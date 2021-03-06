# Generated by CMake

if("${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION}" LESS 2.5)
   message(FATAL_ERROR "CMake >= 2.6.0 required")
endif()
cmake_policy(PUSH)
cmake_policy(VERSION 2.6...3.18)
#----------------------------------------------------------------
# Generated CMake target import file.
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Protect against multiple inclusion, which would fail when already imported targets are added once more.
set(_targetsDefined)
set(_targetsNotDefined)
set(_expectedTargets)
foreach(_expectedTarget pangolin)
  list(APPEND _expectedTargets ${_expectedTarget})
  if(NOT TARGET ${_expectedTarget})
    list(APPEND _targetsNotDefined ${_expectedTarget})
  endif()
  if(TARGET ${_expectedTarget})
    list(APPEND _targetsDefined ${_expectedTarget})
  endif()
endforeach()
if("${_targetsDefined}" STREQUAL "${_expectedTargets}")
  unset(_targetsDefined)
  unset(_targetsNotDefined)
  unset(_expectedTargets)
  set(CMAKE_IMPORT_FILE_VERSION)
  cmake_policy(POP)
  return()
endif()
if(NOT "${_targetsDefined}" STREQUAL "")
  message(FATAL_ERROR "Some (but not all) targets in this export set were already defined.\nTargets Defined: ${_targetsDefined}\nTargets not yet defined: ${_targetsNotDefined}\n")
endif()
unset(_targetsDefined)
unset(_targetsNotDefined)
unset(_expectedTargets)


# Create imported target pangolin
add_library(pangolin SHARED IMPORTED)

set_target_properties(pangolin PROPERTIES
  INTERFACE_INCLUDE_DIRECTORIES "/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX11.1.sdk/System/Library/Frameworks/OpenGL.framework;/usr/local/include;/Users/hpj123/project/project_visnav/thirdparty/eigen;/Users/hpj123/project/project_visnav/thirdparty/Pangolin/include;/Users/hpj123/project/project_visnav/thirdparty/build-Pangolin/src/include"
  INTERFACE_LINK_LIBRARIES "/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX11.1.sdk/System/Library/Frameworks/OpenGL.framework;/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX11.1.sdk/System/Library/Frameworks/OpenGL.framework;/usr/local/lib/libGLEW.dylib;-framework Cocoa;/opt/local/lib/libavcodec.dylib;/opt/local/lib/libavformat.dylib;/opt/local/lib/libavutil.dylib;/opt/local/lib/libswscale.dylib;/opt/local/lib/libavdevice.dylib;/opt/local/lib/libpng.dylib;/opt/local/lib/libz.dylib;/opt/local/lib/libjpeg.dylib;/opt/local/lib/libtiff.dylib;/opt/local/lib/libzstd.dylib;/opt/local/lib/liblz4.dylib"
)

# Import target "pangolin" for configuration "RelWithDebInfo"
set_property(TARGET pangolin APPEND PROPERTY IMPORTED_CONFIGURATIONS RELWITHDEBINFO)
set_target_properties(pangolin PROPERTIES
  IMPORTED_LOCATION_RELWITHDEBINFO "/Users/hpj123/project/project_visnav/thirdparty/build-Pangolin/src/libpangolin.dylib"
  IMPORTED_SONAME_RELWITHDEBINFO "@rpath/libpangolin.dylib"
  )

# This file does not depend on other imported targets which have
# been exported from the same project but in a separate export set.

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
cmake_policy(POP)
