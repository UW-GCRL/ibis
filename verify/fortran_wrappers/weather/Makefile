PY      := /home/hangkai/ibis-port/verify/.venv/bin/python
F2PY    := /home/hangkai/ibis-port/verify/.venv/bin/f2py
FC      := gfortran
CC      := gcc

IBIS_SRC := /home/hangkai/ibis-port/fortran/ibis_2.6b4
MODULE   := weather_wrap
F_WRAP   := $(MODULE).f
F_SUBSET := weather_subset.f

PY_INC      := $(shell $(PY) -c "import sysconfig; print(sysconfig.get_path('include'))")
NP_INC      := $(shell $(PY) -c "import numpy; print(numpy.get_include())")
F2PY_SRC    := $(shell $(PY) -c "import numpy, os; print(os.path.join(os.path.dirname(numpy.__file__), 'f2py', 'src'))")
EXT_SUFFIX  := $(shell $(PY) -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")

SO_FILE := $(MODULE)$(EXT_SUFFIX)

FFLAGS_COV := -c -fprofile-arcs -ftest-coverage -fPIC \
              -fdefault-real-8 -fdefault-double-8 \
              -ffixed-line-length-132 -O0 -w -I$(IBIS_SRC)

FFLAGS_PLAIN := -c -fPIC -O0 -w

CFLAGS := -c -fPIC -O0 \
          -I$(PY_INC) -I$(NP_INC) -I$(F2PY_SRC)

LDFLAGS := -shared -fprofile-arcs

.PHONY: all clean coverage

all: $(SO_FILE)

$(MODULE)module.c $(MODULE)-f2pywrappers.f: $(F_WRAP)
	$(F2PY) -m $(MODULE) $(F_WRAP)

$(F_SUBSET:.f=.o): $(F_SUBSET)
	$(FC) $(FFLAGS_COV) $(F_SUBSET) -o $@

$(MODULE).o: $(F_WRAP)
	$(FC) $(FFLAGS_COV) $(F_WRAP) -o $@

$(MODULE)-f2pywrappers.o: $(MODULE)-f2pywrappers.f
	$(FC) $(FFLAGS_PLAIN) $< -o $@

fortranobject.o: $(F2PY_SRC)/fortranobject.c
	$(CC) $(CFLAGS) $< -o $@

$(MODULE)module.o: $(MODULE)module.c
	$(CC) $(CFLAGS) $< -o $@

$(SO_FILE): $(MODULE).o $(F_SUBSET:.f=.o) $(MODULE)-f2pywrappers.o $(MODULE)module.o fortranobject.o
	$(FC) $(LDFLAGS) $^ -lgcov -o $@
	@echo ""
	@ls -la $@

clean:
	rm -f *.o *.so *.gcno *.gcda *.gcov \
	      $(MODULE)module.c $(MODULE)-f2pywrappers.f .f2py_f2cmap

coverage:
	@if ls *.gcda > /dev/null 2>&1; then \
	    gcov -b $(F_SUBSET) $(F_WRAP); \
	else \
	    echo "No .gcda files yet."; \
	fi
