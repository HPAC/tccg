#!/usr/bin/env python
from distutils.core import setup
import os

OKGREEN = '\033[92m'
FAIL = '\033[91m'
WARNING = '\033[93m'
ENDC = '\033[0m'

setup(name="tccg",
        version="1.0",
        description="Tensor Contraction Code Generator",
        author="Paul Springer",
        author_email="springer@aices.rwth-aachen.de",
        packages=["tccg"],
        scripts=["scripts/tccg"],
        package_data={'tccg': ['Makefile']}
        )

print ""
output = "# "+ FAIL + "IMPORTANT"+ENDC+": execute 'export TCCG_ROOT=%s' #"%(os.path.dirname(os.path.realpath(__file__)))
print '#'*(len(output)-2*len(FAIL)+1)
print output
print '#'*(len(output)-2*len(FAIL)+1)
print ""
