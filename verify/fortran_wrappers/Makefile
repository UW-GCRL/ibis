# =====================================================================
# Hand-rolled F77 -> f2py build with gcov coverage instrumentation.
#
# Why hand-rolled: f2py's `-c` mode uses meson, which does not reliably
# propagate -fprofile-arcs/-ftest-coverage flags to the Fortran compile
# step (and the build dir is cleaned, so .gcno files vanish even when
# the flags are accepted). We need .gcno next to .o so .gcda lands here
# when the .so runs. We get that by compiling each step ourselves.
#
# Pipeline:
#   1. f2py (no -c): generate <module>module.c and <module>-f2pywrappers.f
#   2. gfortran -c:  user .f -> .o with coverage flags (drops .gcno)
#   3. gfortran -c:  f2py aux wrapper .f -> .o (no coverage; boilerplate)
#   4. gcc -c:       numpy/f2py fortranobject.c -> .o
#   5. gcc -c:       generated <module>module.c -> .o
#   6. gfortran -shared: link everything + -lgcov -> .<ext>.so
#
# Run `make coverage` after a verification test to produce .gcov reports.
# =====================================================================

PY      := /home/hangkai/ibis-port/verify/.venv/bin/python
F2PY    := /home/hangkai/ibis-port/verify/.venv/bin/f2py
FC      := gfortran
CC      := gcc

COMSAT_DIR := /home/hangkai/ibis-port/fortran/ibis_2.6b4
MODULE     := thermo_wrap
F_SRC      := $(MODULE).f

# Discover Python/numpy paths at make time so the build is venv-agnostic.
PY_INC      := $(shell $(PY) -c "import sysconfig; print(sysconfig.get_path('include'))")
NP_INC      := $(shell $(PY) -c "import numpy; print(numpy.get_include())")
F2PY_SRC    := $(shell $(PY) -c "import numpy, os; print(os.path.join(os.path.dirname(numpy.__file__), 'f2py', 'src'))")
EXT_SUFFIX  := $(shell $(PY) -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")

SO_FILE := $(MODULE)$(EXT_SUFFIX)

# Float64 promotion matches the JAX port; -fprofile-arcs/-ftest-coverage drop
# .gcno files next to the .o files at compile time.
FFLAGS_COV := -c -fprofile-arcs -ftest-coverage -fPIC \
              -fdefault-real-8 -fdefault-double-8 \
              -ffixed-line-length-132 -O0 -w -I$(COMSAT_DIR)

# Boilerplate f2py-generated F77 wrappers don't need coverage.
FFLAGS_PLAIN := -c -fPIC -O0 -w

CFLAGS := -c -fPIC -O0 \
          -I$(PY_INC) -I$(NP_INC) -I$(F2PY_SRC)

LDFLAGS := -shared -fprofile-arcs

.PHONY: all clean coverage verbose

all: $(SO_FILE)

# Step 1: f2py generates the C wrapper and the aux F77 wrappers.
$(MODULE)module.c $(MODULE)-f2pywrappers.f: $(F_SRC)
	$(F2PY) -m $(MODULE) $(F_SRC)

# Step 2: compile user Fortran with coverage instrumentation.
$(MODULE).o: $(F_SRC)
	$(FC) $(FFLAGS_COV) $(F_SRC) -o $@

# Step 3: compile f2py's aux Fortran wrappers (no coverage; boilerplate).
$(MODULE)-f2pywrappers.o: $(MODULE)-f2pywrappers.f
	$(FC) $(FFLAGS_PLAIN) $< -o $@

# Step 4: compile numpy/f2py's fortranobject.c runtime.
fortranobject.o: $(F2PY_SRC)/fortranobject.c
	$(CC) $(CFLAGS) $< -o $@

# Step 5: compile the generated module C.
$(MODULE)module.o: $(MODULE)module.c
	$(CC) $(CFLAGS) $< -o $@

# Step 6: link.
$(SO_FILE): $(MODULE).o $(MODULE)-f2pywrappers.o $(MODULE)module.o fortranobject.o
	$(FC) $(LDFLAGS) $^ -lgcov -o $@
	@echo ""
	@ls -la $@

verbose:
	@echo "PY_INC      = $(PY_INC)"
	@echo "NP_INC      = $(NP_INC)"
	@echo "F2PY_SRC    = $(F2PY_SRC)"
	@echo "EXT_SUFFIX  = $(EXT_SUFFIX)"
	@echo "SO_FILE     = $(SO_FILE)"

clean:
	rm -f *.o *.so *.gcno *.gcda *.gcov \
	      $(MODULE)module.c $(MODULE)-f2pywrappers.f .f2py_f2cmap

# Run after a verification test has produced .gcda files.
coverage:
	@if ls *.gcda > /dev/null 2>&1; then \
	    gcov -b $(F_SRC); \
	    echo ""; \
	    echo "Per-line coverage in $(F_SRC).gcov"; \
	else \
	    echo "No .gcda files yet -- run verify_thermo.py first."; \
	fi
