Installation
============

Of course, you have several options to install the package: the recommended way to install the package and its dependencies is with `pip`.

## From the latest distributed version

You can get the latest distributed version from the [PyPI](https://pypi.org/project/gw-mbank/) archives.
To do this, you just need to type:

```
pip install gw-mbank
```

This will install the package and its dependencies in your current working python environment.
Unfortunately the name `mbank` wasn't available for the PyPI distribution: this is why the distribution is called `gw-mbank`. I hope this does not confuses you too much, but don't worry: this has no implications for your user experience!

## From source

You can install the package from source. This is useful for development.
To do things in one shot:
```Bash
pip install git+https://github.com/stefanoschmidt1995/mbank
```
This will clone the repo and install the latest (dev) version of `mbank`.

If you want to do things step by step, you can clone the git [repository](https://github.com/stefanoschmidt1995/mbank) and build and install the code manually.
This will build the package and will install it: `pip` will keep track of the dependencies.
These are the steps:

```Bash
git clone git@github.com:stefanoschmidt1995/mbank.git
cd mbank
python setup.py sdist
pip install dist/gw-mbank-*.tar.gz
```

## Build the docs

You can build a local copy of this documentation with:

```Bash
cd mbank
python setup.py build_sphinx
```

The documentation will be built, by default, on `mbank/docs/__build`.



