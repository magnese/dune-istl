# -*- Autoconf -*-
# Process this file with autoconf to produce a configure script.

DUNE_AC_INIT # gets module version from dune.module file
AC_CONFIG_SRCDIR([dune/istl/istlexception.hh])
AC_CONFIG_HEADERS([config.h])

# check all dune-module stuff
DUNE_CHECK_ALL

# set up flags for the automated test system
DUNE_AUTOBUILD_FLAGS

# preset variable to path such that #include <dune/...> works
AC_SUBST([AM_CPPFLAGS], '$(DUNE_CPPFLAGS) -I$(top_srcdir)')
AC_SUBST([AM_LDFLAGS], '$(DUNE_LDFLAGS) $(DUNE_LIBS)')

# write output
AC_CONFIG_FILES([Makefile
    cmake/Makefile
    cmake/modules/Makefile
    doc/doxygen/Makefile
    doc/doxygen/Doxyfile
    doc/Makefile
    dune/istl/eigenvalue/Makefile
    dune/istl/paamg/test/Makefile
    dune/istl/paamg/Makefile
    dune/istl/tutorial/Makefile
    dune/istl/test/Makefile
    dune/istl/Makefile
    dune/Makefile
    lib/Makefile
    m4/Makefile
    dune-istl.pc])
AC_OUTPUT

DUNE_SUMMARY_ALL
