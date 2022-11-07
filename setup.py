import setuptools
import sys

try:
	from sphinx.setup_command import BuildDoc
	cmdclass = {'build_sphinx': BuildDoc} #to build with sphinx
except ImportError:
	if sys.argv[1] == 'build_sphinx':
		raise ImportErorr("sphinx modules not found: impossibile to build the documents.\nTry: pip install -r docs/requirements.txt")

required_packages =['scipy', 'numpy', 'matplotlib',
	'python-ligo-lw==1.7.1', 'lalsuite>=6.70', 'tqdm', 'ray'] #the dependencies are fucked up, for some reason
#required_packages =[]

with open("README.md", "r") as fh:
	long_description = fh.read()

setuptools.setup(
	name="gw-mbank",
	version="0.1.0",
	author="Stefano Schmidt",
	author_email="stefanoschmidt1995@gmail.com",
	description="Metric bank generation for gravitational waves data analysis",
	long_description=long_description,
	long_description_content_type="text/markdown",
	url="https://github.com/stefanoschmidt1995/mbank",
	packages=setuptools.find_packages(),
	license = 'GNU GENERAL PUBLIC LICENSE v3',
	classifiers=[
		"Programming Language :: Python :: 3",
		"Operating System :: OS Independent",
	],
	scripts = ["bin/mbank_run", "bin/mbank_injfile", "bin/mbank_mcmc", 
		"bin/mbank_place_templates", "bin/mbank_merge","bin/mbank_validate_metric", "bin/mbank_injections"],
	python_requires='>=3.7',
	install_requires=required_packages,
	command_options={
        'build_sphinx': {
            'source_dir': ('setup.py', 'docs'),
            'build_dir': ('setup.py', 'docs/__build'),
            }},
)

