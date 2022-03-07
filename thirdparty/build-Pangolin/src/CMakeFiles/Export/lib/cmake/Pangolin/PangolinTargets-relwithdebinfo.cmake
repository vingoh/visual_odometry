#----------------------------------------------------------------
# Generated CMake target import file for configuration "RelWithDebInfo".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "pangolin" for configuration "RelWithDebInfo"
set_property(TARGET pangolin APPEND PROPERTY IMPORTED_CONFIGURATIONS RELWITHDEBINFO)
set_target_properties(pangolin PROPERTIES
  IMPORTED_LOCATION_RELWITHDEBINFO "/usr/local/lib/libpangolin.dylib"
  IMPORTED_SONAME_RELWITHDEBINFO "@rpath/libpangolin.dylib"
  )

list(APPEND _IMPORT_CHECK_TARGETS pangolin )
list(APPEND _IMPORT_CHECK_FILES_FOR_pangolin "/usr/local/lib/libpangolin.dylib" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
