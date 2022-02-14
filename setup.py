import setuptools
import warnings
import sys

try:
	from sphinx.setup_command import BuildDoc
	cmdclass = {'build_sphinx': BuildDoc} #to build with sphinx
except ImportError:
	if sys.argv[1] == 'build_sphinx': warnings.warn("sphinx module not found: impossibile to build the documents")
	pass

required_packages =['scipy', 'numpy>=1.21.5', 'matplotlib>=3.5.1',
	'python-ligo-lw>=1.7.1', 'lalsuite>=6.70', 'tqdm>=4.62.3', 'ray>=1.0.0'] #the dependencies are fucked up, for some reason
required_packages =[]

with open("README.md", "r") as fh:
	long_description = fh.read()

setuptools.setup(
	name="mbank",
	version="0.0.1",
	author="Stefano Schmidt",
	author_email="stefanoschmidt1995@gmail.com",
	description="Metric bank generation for gravitational waves data analysis",
	long_description=long_description,
	long_description_content_type="text/markdown",
	url="https://github.com/stefanoschmidt1995/mbank",
	packages=setuptools.find_packages(),
	licence = 'GNU GENERAL PUBLIC LICENSE v3',
	classifiers=[
		"Programming Language :: Python :: 3",
		"Operating System :: OS Independent",
	],
	scripts = ["bin/mbank_run", "bin/mbank_injfile", "bin/mbank_place_templates", "bin/mbank_injections"],
	python_requires='>=3.7',
	install_requires=required_packages,
	command_options={
        'build_sphinx': {
            'source_dir': ('setup.py', 'docs'),
            'build_dir': ('setup.py', 'docs/__build'),
            }},
)

