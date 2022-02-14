Installation
============

The recommended way to install the package and its dependencies is with `pip`.
You have several options to install the package.

## From the latest distributed version

You can get the latest distributed version from the [PyPI](https://pypi.org/project/mbank/) archives.
To do this, you just need to type:

```
pip install mbank
```

This will install the package and its dependencies in your current working python environment.

## From source

You can install the package from source. This is useful for development.
To do this you need to clone the git [repository](https://github.com/stefanoschmidt1995/mbank) and to build and install the code manually. A handy makefile will help you to do that.

These are the steps:


```Bash
git clone git@github.com:stefanoschmidt1995/mbank.git
cd mbank
make install
```

This will build the package and will install it: `pip` will keep track of the dependencies.

If you want to do things by hands, or to keep track of the dependencies on your own, you can type:

```Bash
git clone git@github.com:stefanoschmidt1995/mbank.git
cd mbank
python setup.py sdist
pip install -r requirements.txt
pip install dist/mbank-0.0.1.tar.gz
```

## Build the docs

You can build a local copy of this documentation with:

```Bash
cd mbank
python setup.py build_sphinx
```

The documentation will be built, by default, on `mbank/docs/__build`.



