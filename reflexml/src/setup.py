import importlib
import os
from pip.req import parse_requirements
from setuptools import setup
from setuptools import find_packages
import sys

PACKAGE_NAME = 'reflexml'
PACKAGE_ROOT = 'reflexml'
ROOT_INIT = os.path.join(PACKAGE_ROOT, '__init__.py')

exec([ln for ln in open(ROOT_INIT, 'r').readlines() if '__version__' in ln][-1])

PACKAGE_VERSION = __version__

DELICATE_LIBS = ['cv2', 'dlib']

for lib in DELICATE_LIBS:
    try:
        importlib.import_module(lib)
    except:
        raise ImportError('Module: {} required for usage of {}'.format(lib, PACKAGE_NAME))

install_reqs = parse_requirements('requirements.txt', session=False)
reqs = [str(ir.req) for ir in install_reqs]

setup(
    name='reflexml',
    version=PACKAGE_VERSION,
    description='Library for building Machine Learning products in the Vizzario Stack',
    author='Luke de Oliveira',
    author_email='lukedeo@ldo.io',
    install_requires=reqs,
    packages=find_packages()
)


