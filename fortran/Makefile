# =============================================================================
# Makefile  --  IBIS modernisation build system
#
# Requires: gfortran >= 8 (F2008 support)
#
# Targets:
#   make all          -- compile all modernised modules (default)
#   make test         -- compile and run all test drivers
#   make test_phase1  -- compile and run Phase 1 tests only
#   make test_phase2  -- compile and run Phase 2 tests only
#   make test_phase3  -- compile and run Phase 3 tests only
#   make clean        -- remove all build artefacts
#
# Precision switch:
#   make PREC=sp      -- single-precision regression build (-DIBIS_SP)
#   make              -- double-precision production build (default)
#
# =============================================================================

FC      := gfortran
FFLAGS  := -Wall -Wextra -fcheck=all -std=f2008 -cpp -g -fbacktrace

# Working precision: default double; override with PREC=sp
ifeq ($(PREC),sp)
  FFLAGS += -DIBIS_SP
endif

# NetCDF Fortran-3 include path (for ies-io.f90 which uses netcdf.inc)
# Override with: make NETCDF_INC=/path/to/netcdf/include
NETCDF_INC ?= $(shell pkg-config --variable=includedir netcdf-fortran 2>/dev/null || echo /opt/homebrew/include)
FFLAGS += -I$(NETCDF_INC)

# Build output directories
MODDIR := build/modules
OBJDIR := build/objects

# Module output + search path (gfortran: -J writes .mod files to dir and
# adds it to the search path)
FFLAGS += -J$(MODDIR) -I$(MODDIR)

# =============================================================================
# Phase 1: foundation modules (no dependencies on other IBIS modules)
# =============================================================================
PHASE1_SRCS := src/modules/mod_precision.f90   \
               src/modules/mod_dimensions.f90  \
               src/modules/mod_constants.f90

PHASE1_OBJS := $(patsubst src/modules/%.f90,$(OBJDIR)/%.o,$(PHASE1_SRCS))

# =============================================================================
# Phase 2: state modules (depend on Phase 1 foundations)
# =============================================================================
PHASE2_SRCS := src/modules/mod_bgc_params.f90    \
               src/modules/mod_soil_texture.f90   \
               src/modules/mod_age_class.f90      \
               src/modules/mod_diagnostics.f90    \
               src/modules/mod_pft_params.f90     \
               src/modules/mod_thermo_funcs.f90

PHASE2_OBJS := $(patsubst src/modules/%.f90,$(OBJDIR)/%.o,$(PHASE2_SRCS))

# =============================================================================
# Phase 3: large-domain state modules (depend on Phase 1 and Phase 2)
# =============================================================================
PHASE3_SRCS := src/modules/mod_snow_state.f90      \
               src/modules/mod_hydrology.f90        \
               src/modules/mod_carbon_pools.f90     \
               src/modules/mod_atm_state.f90        \
               src/modules/mod_boundary_climate.f90 \
               src/modules/mod_soil_state.f90       \
               src/modules/mod_canopy_fluxes.f90    \
               src/modules/mod_work.f90             \
               src/modules/mod_summary.f90          \
               src/modules/mod_canopy_params.f90    \
               src/modules/mod_veg_dynamics.f90     \
               src/modules/mod_run_params.f90

PHASE3_OBJS := $(patsubst src/modules/%.f90,$(OBJDIR)/%.o,$(PHASE3_SRCS))

# NetCDF link flags (Fortran + C libraries)
NETCDF_LIB ?= $(shell pkg-config --libs netcdf-fortran 2>/dev/null || echo -L/opt/homebrew/lib -lnetcdff -lnetcdf)

# =============================================================================
# Aggregate targets
# =============================================================================
.PHONY: all test test_phase1 test_phase2 test_phase3 phase4 ibis ibis_global ibis_f77 clean

all: $(PHASE1_OBJS) $(PHASE2_OBJS) $(PHASE3_OBJS)

test: test_phase1 test_phase2 test_phase3

# Link all Phase 1-4 objects into the ibis executable.
# Prerequisite is the 'phase4' phony target (builds all objects);
# object variables are listed explicitly in the recipe because they are
# defined later in the file (after the Phase 4 source lists).
ibis: phase4
	$(FC) $(FFLAGS) $(ALL_MOD_OBJS) $(PHASE4_OBJS) $(NETCDF_LIB) -o $@

# Build the original F77 binary with gfortran in legacy mode (replaces g77).
# Compile order follows the original ibis_2.6b4/makefile ibis target.
F77_SRCS := ibis_2.6b4/utilities.f  \
            ibis_2.6b4/math.f        \
            ibis_2.6b4/ies-io.f      \
            ibis_2.6b4/readpars.f    \
            ibis_2.6b4/initial.f     \
            ibis_2.6b4/io.f          \
            ibis_2.6b4/diagnostics.f \
            ibis_2.6b4/climate.f     \
            ibis_2.6b4/radiation.f   \
            ibis_2.6b4/snow.f        \
            ibis_2.6b4/soil.f        \
            ibis_2.6b4/weather.f     \
            ibis_2.6b4/vegetation.f  \
            ibis_2.6b4/biogeochem.f  \
            ibis_2.6b4/physiology.f  \
            ibis_2.6b4/canopy.f      \
            ibis_2.6b4/stats.f       \
            ibis_2.6b4/main.f

ibis_f77: $(F77_SRCS)
	gfortran -std=legacy -ffixed-form -ffixed-line-length-132 -funroll-loops \
	  -I$(NETCDF_INC) $(F77_SRCS) $(NETCDF_LIB) -o $@

clean:
	rm -rf build build_global tests/test_phase1 tests/test_phase2 tests/test_phase3 ibis ibis_global ibis_f77

# =============================================================================
# Phase 1 object rules (explicit to encode inter-module dependencies)
# =============================================================================

$(OBJDIR)/mod_precision.o: src/modules/mod_precision.f90 | $(OBJDIR) $(MODDIR)
	$(FC) $(FFLAGS) -c $< -o $@

$(OBJDIR)/mod_dimensions.o: src/modules/mod_dimensions.f90 \
                             $(OBJDIR)/mod_precision.o | $(OBJDIR)
	$(FC) $(FFLAGS) -c $< -o $@

$(OBJDIR)/mod_constants.o: src/modules/mod_constants.f90 \
                            $(OBJDIR)/mod_precision.o  \
                            $(OBJDIR)/mod_dimensions.o | $(OBJDIR)
	$(FC) $(FFLAGS) -c $< -o $@

# =============================================================================
# Phase 2 object rules
# =============================================================================

$(OBJDIR)/mod_bgc_params.o: src/modules/mod_bgc_params.f90 \
                             $(OBJDIR)/mod_precision.o | $(OBJDIR)
	$(FC) $(FFLAGS) -c $< -o $@

$(OBJDIR)/mod_soil_texture.o: src/modules/mod_soil_texture.f90 \
                               $(OBJDIR)/mod_precision.o | $(OBJDIR)
	$(FC) $(FFLAGS) -c $< -o $@

$(OBJDIR)/mod_age_class.o: src/modules/mod_age_class.f90 \
                            $(OBJDIR)/mod_precision.o | $(OBJDIR)
	$(FC) $(FFLAGS) -c $< -o $@

$(OBJDIR)/mod_diagnostics.o: src/modules/mod_diagnostics.f90 | $(OBJDIR)
	$(FC) $(FFLAGS) -c $< -o $@

$(OBJDIR)/mod_pft_params.o: src/modules/mod_pft_params.f90 \
                             $(OBJDIR)/mod_precision.o  \
                             $(OBJDIR)/mod_dimensions.o | $(OBJDIR)
	$(FC) $(FFLAGS) -c $< -o $@

$(OBJDIR)/mod_thermo_funcs.o: src/modules/mod_thermo_funcs.f90 \
                               $(OBJDIR)/mod_precision.o  \
                               $(OBJDIR)/mod_constants.o | $(OBJDIR)
	$(FC) $(FFLAGS) -c $< -o $@

# =============================================================================
# Phase 3 object rules
# All depend on mod_precision (wp) and mod_dimensions (dimension constants).
# mod_carbon_pools additionally uses mod_dimensions npft.
# mod_canopy_params additionally uses mod_dimensions nband.
# mod_veg_dynamics additionally uses mod_dimensions nsoilay.
# =============================================================================

$(OBJDIR)/mod_snow_state.o: src/modules/mod_snow_state.f90 \
                             $(OBJDIR)/mod_precision.o  \
                             $(OBJDIR)/mod_dimensions.o | $(OBJDIR)
	$(FC) $(FFLAGS) -c $< -o $@

$(OBJDIR)/mod_hydrology.o: src/modules/mod_hydrology.f90 \
                            $(OBJDIR)/mod_precision.o  \
                            $(OBJDIR)/mod_dimensions.o | $(OBJDIR)
	$(FC) $(FFLAGS) -c $< -o $@

$(OBJDIR)/mod_carbon_pools.o: src/modules/mod_carbon_pools.f90 \
                               $(OBJDIR)/mod_precision.o  \
                               $(OBJDIR)/mod_dimensions.o | $(OBJDIR)
	$(FC) $(FFLAGS) -c $< -o $@

$(OBJDIR)/mod_atm_state.o: src/modules/mod_atm_state.f90 \
                            $(OBJDIR)/mod_precision.o  \
                            $(OBJDIR)/mod_dimensions.o | $(OBJDIR)
	$(FC) $(FFLAGS) -c $< -o $@

$(OBJDIR)/mod_boundary_climate.o: src/modules/mod_boundary_climate.f90 \
                                   $(OBJDIR)/mod_precision.o  \
                                   $(OBJDIR)/mod_dimensions.o | $(OBJDIR)
	$(FC) $(FFLAGS) -c $< -o $@

$(OBJDIR)/mod_soil_state.o: src/modules/mod_soil_state.f90 \
                             $(OBJDIR)/mod_precision.o  \
                             $(OBJDIR)/mod_dimensions.o | $(OBJDIR)
	$(FC) $(FFLAGS) -c $< -o $@

$(OBJDIR)/mod_canopy_fluxes.o: src/modules/mod_canopy_fluxes.f90 \
                                $(OBJDIR)/mod_precision.o  \
                                $(OBJDIR)/mod_dimensions.o | $(OBJDIR)
	$(FC) $(FFLAGS) -c $< -o $@

$(OBJDIR)/mod_work.o: src/modules/mod_work.f90 \
                       $(OBJDIR)/mod_precision.o  \
                       $(OBJDIR)/mod_dimensions.o | $(OBJDIR)
	$(FC) $(FFLAGS) -c $< -o $@

$(OBJDIR)/mod_summary.o: src/modules/mod_summary.f90 \
                          $(OBJDIR)/mod_precision.o  \
                          $(OBJDIR)/mod_dimensions.o | $(OBJDIR)
	$(FC) $(FFLAGS) -c $< -o $@

$(OBJDIR)/mod_canopy_params.o: src/modules/mod_canopy_params.f90 \
                                $(OBJDIR)/mod_precision.o  \
                                $(OBJDIR)/mod_dimensions.o | $(OBJDIR)
	$(FC) $(FFLAGS) -c $< -o $@

$(OBJDIR)/mod_veg_dynamics.o: src/modules/mod_veg_dynamics.f90 \
                               $(OBJDIR)/mod_precision.o  \
                               $(OBJDIR)/mod_dimensions.o | $(OBJDIR)
	$(FC) $(FFLAGS) -c $< -o $@

$(OBJDIR)/mod_run_params.o: src/modules/mod_run_params.f90 \
                             $(OBJDIR)/mod_precision.o  \
                             $(OBJDIR)/mod_dimensions.o | $(OBJDIR)
	$(FC) $(FFLAGS) -c $< -o $@

# =============================================================================
# Phase 4: converted physics and I/O routines (F77 -> F2008)
# Each .f90 depends on all module .o files that it USEs at compile time.
# Object rules use an explicit list of module prerequisites so that parallel
# make (-j) orders correctly.
#
# ALL_MOD_OBJS collects every module object so physics rules can depend on
# the full set without spelling them out per-file.
# =============================================================================

ALL_MOD_OBJS := $(PHASE1_OBJS) $(PHASE2_OBJS) $(PHASE3_OBJS)

PHYOBJDIR := build/objects
PHYSSRCDIR := src/physics
IOSRCDIR  := src/io

PHASE4_PHYS_SRCS := $(PHYSSRCDIR)/utilities.f90  \
                    $(PHYSSRCDIR)/math.f90         \
                    $(PHYSSRCDIR)/snow.f90         \
                    $(PHYSSRCDIR)/climate.f90      \
                    $(PHYSSRCDIR)/radiation.f90    \
                    $(PHYSSRCDIR)/weather.f90      \
                    $(PHYSSRCDIR)/vegetation.f90   \
                    $(PHYSSRCDIR)/biogeochem.f90   \
                    $(PHYSSRCDIR)/soil.f90         \
                    $(PHYSSRCDIR)/physiology.f90   \
                    $(PHYSSRCDIR)/canopy.f90       \
                    $(PHYSSRCDIR)/stats.f90

PHASE4_IO_SRCS := $(IOSRCDIR)/diagnostics.f90    \
                  $(IOSRCDIR)/readpars.f90         \
                  $(IOSRCDIR)/initial.f90          \
                  $(IOSRCDIR)/io.f90               \
                  $(IOSRCDIR)/ies-io.f90

PHASE4_MAIN_SRC := src/main.f90

PHASE4_PHYS_OBJS := $(patsubst $(PHYSSRCDIR)/%.f90,$(OBJDIR)/%.o,$(PHASE4_PHYS_SRCS))
PHASE4_IO_OBJS   := $(patsubst $(IOSRCDIR)/%.f90,$(OBJDIR)/%.o,$(PHASE4_IO_SRCS))
PHASE4_MAIN_OBJ  := $(OBJDIR)/main.o

PHASE4_OBJS := $(PHASE4_PHYS_OBJS) $(PHASE4_IO_OBJS) $(PHASE4_MAIN_OBJ)

.PHONY: phase4

phase4: $(PHASE4_OBJS)

# Generic rule: any Phase 4 physics/IO .f90 file depends on ALL module objects
$(PHASE4_PHYS_OBJS): $(OBJDIR)/%.o: $(PHYSSRCDIR)/%.f90 $(ALL_MOD_OBJS) | $(OBJDIR)
	$(FC) $(FFLAGS) -c $< -o $@

$(PHASE4_IO_OBJS): $(OBJDIR)/%.o: $(IOSRCDIR)/%.f90 $(ALL_MOD_OBJS) | $(OBJDIR)
	$(FC) $(FFLAGS) -c $< -o $@

$(PHASE4_MAIN_OBJ): src/main.f90 $(ALL_MOD_OBJS) $(PHASE4_PHYS_OBJS) $(PHASE4_IO_OBJS) | $(OBJDIR)
	$(FC) $(FFLAGS) -c $< -o $@

# =============================================================================
# Test drivers
# =============================================================================

tests/test_phase1: tests/test_phase1.f90 $(PHASE1_OBJS) | $(OBJDIR)
	$(FC) $(FFLAGS) $^ -o $@

test_phase1: tests/test_phase1
	@echo ''
	@echo '=== Running Phase 1 tests ==='
	./tests/test_phase1
	@echo ''

tests/test_phase2: tests/test_phase2.f90 $(PHASE1_OBJS) $(PHASE2_OBJS) | $(OBJDIR)
	$(FC) $(FFLAGS) $^ -o $@

test_phase2: tests/test_phase2
	@echo ''
	@echo '=== Running Phase 2 tests ==='
	./tests/test_phase2
	@echo ''

tests/test_phase3: tests/test_phase3.f90 $(PHASE1_OBJS) $(PHASE2_OBJS) $(PHASE3_OBJS) | $(OBJDIR)
	$(FC) $(FFLAGS) $^ -o $@

test_phase3: tests/test_phase3
	@echo ''
	@echo '=== Running Phase 3 tests ==='
	./tests/test_phase3
	@echo ''

# =============================================================================
# Directory creation (order-only prerequisites)
# =============================================================================
$(OBJDIR) $(MODDIR):
	mkdir -p $@

# =============================================================================
# Global domain binary (ibis_global)
# Compiled into build_global/ with -DGLOBAL → npoi=58920 in mod_dimensions.f90.
# Defined AFTER PHASE4_*_SRCS so patsubst has the source lists available.
# =============================================================================
GMODDIR := build_global/modules
GOBJDIR := build_global/objects

# Strip the MODDIR flags from FFLAGS and add global equivalents
GFLAGS  := $(filter-out -J$(MODDIR) -I$(MODDIR),$(FFLAGS)) \
           -DGLOBAL -J$(GMODDIR) -I$(GMODDIR)

GPHASE1_OBJS := $(patsubst src/modules/%.f90,$(GOBJDIR)/%.o,$(PHASE1_SRCS))
GPHASE2_OBJS := $(patsubst src/modules/%.f90,$(GOBJDIR)/%.o,$(PHASE2_SRCS))
GPHASE3_OBJS := $(patsubst src/modules/%.f90,$(GOBJDIR)/%.o,$(PHASE3_SRCS))
GALL_MOD_OBJS := $(GPHASE1_OBJS) $(GPHASE2_OBJS) $(GPHASE3_OBJS)

GPHASE4_PHYS_OBJS := $(patsubst $(PHYSSRCDIR)/%.f90,$(GOBJDIR)/%.o,$(PHASE4_PHYS_SRCS))
GPHASE4_IO_OBJS   := $(patsubst $(IOSRCDIR)/%.f90,  $(GOBJDIR)/%.o,$(PHASE4_IO_SRCS))
GPHASE4_MAIN_OBJ  := $(GOBJDIR)/main.o
GPHASE4_OBJS      := $(GPHASE4_PHYS_OBJS) $(GPHASE4_IO_OBJS) $(GPHASE4_MAIN_OBJ)

$(GOBJDIR) $(GMODDIR):
	mkdir -p $@

$(GPHASE1_OBJS): $(GOBJDIR)/%.o: src/modules/%.f90 | $(GOBJDIR) $(GMODDIR)
	$(FC) $(GFLAGS) -c $< -o $@

$(GPHASE2_OBJS): $(GOBJDIR)/%.o: src/modules/%.f90 $(GPHASE1_OBJS) | $(GOBJDIR)
	$(FC) $(GFLAGS) -c $< -o $@

$(GPHASE3_OBJS): $(GOBJDIR)/%.o: src/modules/%.f90 $(GPHASE1_OBJS) $(GPHASE2_OBJS) | $(GOBJDIR)
	$(FC) $(GFLAGS) -c $< -o $@

$(GPHASE4_PHYS_OBJS): $(GOBJDIR)/%.o: $(PHYSSRCDIR)/%.f90 $(GALL_MOD_OBJS) | $(GOBJDIR)
	$(FC) $(GFLAGS) -c $< -o $@

$(GPHASE4_IO_OBJS): $(GOBJDIR)/%.o: $(IOSRCDIR)/%.f90 $(GALL_MOD_OBJS) | $(GOBJDIR)
	$(FC) $(GFLAGS) -c $< -o $@

$(GPHASE4_MAIN_OBJ): src/main.f90 $(GALL_MOD_OBJS) $(GPHASE4_PHYS_OBJS) $(GPHASE4_IO_OBJS) | $(GOBJDIR)
	$(FC) $(GFLAGS) -c $< -o $@

ibis_global: $(GALL_MOD_OBJS) $(GPHASE4_OBJS)
	$(FC) $(GFLAGS) $(GALL_MOD_OBJS) $(GPHASE4_OBJS) $(NETCDF_LIB) -o $@
