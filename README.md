# mbank
`mbank` is a code for fast Gravitational Waves template bank generation. It creates a bank of binary black hole (BBH) systems. It is very handy for generating precessing and eccentric banks.

If you want more details, you can take a look at the [documentation](https://mbank.readthedocs.io/en/latest/) or at the [paper](https://arxiv.org/abs/2302.00436).
Otherwise, you can keep reading and learn the essentials below.

## How it works
In order to search for a Binary Black Hole (BBH) signal, one needs to come up with a set of templates (a.k.a. bank), signals that will be searched in the noisy data from the interferometers.
Generating a [bank](https://journals.aps.org/prd/abstract/10.1103/PhysRevD.80.104014) is a tedious work, requiring to place huge number of templates so that their mutual distance is as constant as possible. Unfortunately the computation of such distance is highly expensive and, if we want to expand the parameter space covered by the searches, we are **very** interested to get a reliable bank covering an high dimensional space at a decent cost.

A first step consists in introducing a cheap approximation to the distance between templates. The approximation relies on replacing the complicated distance with a _metric_ distance (i.e. a bilinear form). This however can still give problems, as computing the metric can still be quite expensive, especially in high dimensions.

An appealing alternative to computing distances is to sample templates from a suitable distribution, which ensures that the templates are as equally spaced as possible. This is the "uniform" distribution across the parameter space (which of course is not uniform in the space coordinates) and it is characterized by a volume element (equal to the square root of the metric determinant, as standard in differential geometry).

`mbank` does all of this. It is able to compute the metric and it employs a normalizing flow (machine learning model) to estimate the volume element and to sample from the parameter space.

The bank generation algorithm works in 4+1 steps:

1. Defining a metric approximation
2. Generating a dataset, where the volume element is computed at many points in the space
3. Training a normalizing flow to sample from the space and to quickly estimate the volume element
4. Placing the templates by sampling from the normalizing flow
5. Validate the bank by means of injections

`mbank` is the code that does all of this for you!

## How to install

To install the latest [released](https://pypi.org/project/gw-mbank/) version (no release has been done yet):

```Bash
pip install gw-mbank
```

To intall the latest version in the github repository, you can type:

```Bash
pip install git+https://github.com/stefanoschmidt1995/mbank
```

Otherwise, you can clone the repo, build a distribution and install the package:

```Bash
git clone git@github.com:stefanoschmidt1995/mbank.git
cd mbank
python setup.py sdist
pip install dist/gw-mbank*.tar.gz
```

This will install the source code as well as some executables that makes the bank generation easier (plus the dependencies).
See also the [docs](https://mbank.readthedocs.io/en/latest/usage/install.html).

## How to use

To generate a bank you can use several executables. Make sure you have a PSD file (either in csv file either in ligo xml format).
You will need to make several choices on how your bank looks like, such as:
- The BBH variables that you want to vary within the bank (`--variable_format` parameter)
- The minimum match (`--mm`), that controls the average spacing between templates
- The range of physical parameters you want to include in the bank (note that the spins are _always_ expressed in spherical coordinates)
- Low and high frequency for the match/metric computation (`--f-min` and `--f-max`)
- The WF FD approximant (it must be `lal`)
- The architecture of the normalizing flow, as well as several choices on how to train it
- The placing method `--placing-method` for the templates in each tile ('geometric', 'stochastic' or 'random'). The 'random' method is recommended (and the only fully validated).

If you don't have a favourite PSD, you can download one with `wget https://dcc.ligo.org/public/0165/T2000012/002/aligo_O3actual_H1.txt`.

After you made all the choices, the bank generation happens with three different executables:

1. `mbank_generate_flow_dataset`
2. `mbank_train_flow`
3. `mbank_place_templates`

All of those choices can be specified through options to the available executables. If you don't feel like typing all the options every time, you can add them to a text file `myBank.ini` and pass it to the relevnt command: it will figure everything out by itself. You can find some example [ini files](https://github.com/stefanoschmidt1995/mbank/tree/master/examples) in the repo.

Once you have an `ini` file, you can [generate](https://mbank.readthedocs.io/en/latest/usage/bank_generation.html) a template bank with: 

```Bash
mbank_generate_flow_dataset myBank.ini
mbank_train_flow myBank.ini
mbank_place_templates myBank.ini
```

You can also generate an injection file (sampling injections from the normalizing flow) and [validate](https://mbank.readthedocs.io/en/latest/usage/injections.html) your template bank. This is done by computing the fitting factor for each injection:

```Bash
mbank_injfile myBank.ini
mbank_injections myBank.ini
```

As you see, the same file can be used for different commands: each command will just ignore any option not relevant for it.

You can read a lot more details in the [documentation page](https://mbank.readthedocs.io/en/latest/).

## Contacts

Fore more info, or just to say hello, you can contact me: [stefanoschmidt1995@gmail.com](mailto:stefanoschmidt1995@gmail.com).




















