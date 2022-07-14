# mbank
`mbank` is a code for fast Gravitational Waves template bank generation. It creates a bank of binary black hole (BBH) systems. It is very handy for generating precessing and eccentric banks.

If you want more details, you can take a look at the [documentation](https://mbank.readthedocs.io/en/latest/).
Otherwise, you can keep reading and learn the essentials below.

## How it works
Write something...

## How to install

To install the latest released verion (no release has been done yet):

```Bash
pip install mbank
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
pip install dist/mbank-0.0.1.tar.gz
```
This will install the source code as well as some executables that makes the bank generation easier (plus the dependencies).

P.s. Make sure you installed the required packages `pip install -r ./requirements.txt`. In a future version, this will be made a bit more simple and the dependencies will be tracked authomatically by `pip`.

## How to use

To generate a bank you can use the executable `mbank_run`. Make sure you have a PSD file (either in csv file either in ligo xml format).
You will need to choose:
- The BBH variables that you want to vary within the bank (`--variable_format` parameter)
- The minimum match (`--MM`), that controls the average distance between templates
- The range of physical parameters you want to include in the bank (note that the spins are _always_ expressed in spherical coordinates)
- Low and high frequency for the match/metric computation (`--f-min` and `--f-max`)
- The WF FD approximant (it must be lal)
- Maximum number of templates in each tile: this tunes the hierarchical tiling (`--template-in-tile` argument)
- A coarse grid for tiling: the tiling can be parallelized and performed independently on each split (`--grid-size` argument)
- The placing method `--placing-method` for the templates in each tile ('geometric', 'stochastic', 'pure_stochastic', 'uniform', 'iterative'). The geometric method is recommended.

An example command to generate a simple precessing bank with precession placed only on one BH is:
```Bash
mbank_run \
	--run-name myFirstBank \
	--variable-format Mq_s1z_s2z \
	--grid-size 1,1,2,2 \
	--mm 0.97 \
	--tile-tolerance 0.5 \
	--max-depth 10 \
	--psd examples/aligo_O3actual_H1.txt --asd \
	--f-min 15 --f-max 1024 \
	--mtot-range 20 75 \
	--q-range 1 5 \
	--s1-range 0.0 0.99 \
	--s2-range -0.99 0.99 \
	--plot \
	--placing-method random \
	--livepoints 100 \
	--approximant IMRPhenomPv2 \
	--use-ray 
```
To know more information about the available options type:
```Bash
mbank_run --help
```
This is how the output bank look like:

![](docs/img/bank_README.png)

You can also use the metric to estimate the fitting factor for a bunch of injections: 

```Bash
mbank_injections \
	--n-injs 10000 \
	--N-neigh-templates 100 \
	--variable-format Mq_s1z_s2z \
	--tiling-file out_myFirstBank/tiling_myFirstBank.npy \
	--bank-file out_myFirstBank/bank_myFirstBank.xml.gz \
	--psd examples/aligo_O3actual_H1.txt --asd \
	--approximant IMRPhenomPv2 \
	--f-min 15 --f-max 1024 \
	--plot
```

If you specify the `--full-match` option, the match will be recomputed without a metric approximation: in this case, you want to speed things up with something like `--use-ray` and `--cache` (if you have enough memory).
You can also throw some injection chosen from a file: you just need to set an input xml injection file with the `--inj-file` option.

Here's the injection recovery:

![](docs/img/injections_README.png)

If you don't feel like typing all the options every time, you can add them to a text file `myFirstBank.ini` and pass it to the command: it will figure out by itself. You can find some example [ini files](https://github.com/stefanoschmidt1995/mbank/tree/master/examples) in the repo. To run them:

```Bash
mbank_run myFirstBank.ini
mbank_injections myFirstBank.ini
```

As you see, the same file can be used for different commands: each command will just ignore any option not relevant for it.


## Contacts

Fore more info, or just to say hello, you can contact me: [stefanoschmidt1995@gmail.com](mailto:stefanoschmidt1995@gmail.com).




















