PYTHON=python3
RM=/usr/bin/rm

.PHONY: all clean test test-coverage

all:
	$(PYTHON) setup.py build_ext --inplace

clean:
	$(RM) -r build
	$(RM) util_pyx/util.c
	$(RM) util_pyx/util*.so

test:
	$(PYTHON)  -m nose2.__main__  -v

test-coverage:
	coverage run  -m nose2.__main__  -v
