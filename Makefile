PYTHON=python3
RM=/usr/bin/rm



all:
	$(PYTHON) setup.py build_ext --inplace

clean:
	$(RM) -r build
	$(RM) util_pyx/util.c
	$(RM) util_pyx/util*.so
