import setuptools

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
	scripts = ["bin/run_mbank", "bin/run_bank_injs", "bin/mbank_place_templates", "bin/mbank_injections"],
	python_requires='>=3.6',
)

