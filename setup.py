import os
import warnings
from setuptools import setup, find_packages

with open(os.path.join(os.path.abspath(os.path.dirname(__file__)), 'README.rst'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='tfrt',
    version='0.1.dev',
    classifiers=[
        'Development Status :: 4 - Beta',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.6',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Astronomy'],
    packages=find_packages(),
    include_package_data=True,
    python_requires='>=3.6',
    install_requires=[
        'numpy',
        'matplotlib',
        'tqdm',
        'packaging'],
    extras_require={
        "tensorflow": ["tensorflow>=2.3.0"]},
    url='https://github.com/henrysky/simple_tf_raytracing',
    project_urls={
        "Bug Tracker": "https://github.com/henrysky/simple_tf_raytracing/issues",
        "Documentation": "https://github.com/henrysky/simple_tf_raytracing",
        "Source Code": "https://github.com/henrysky/simple_tf_raytracing",
    },
    license='MIT',
    author='Henry Leung',
    author_email='henrysky.leung@utoronto.ca',
    description='Simple Ray-Tracing with Tensorflow',
    long_description=long_description
)

# check if user has tf installed as they are not strict requirements
try:
    import tensorflow
except ImportError:
    warnings.warn("Tensorflow not found, please install tensorflow or tensorflow_gpu or tensorflow_cpu manually!")