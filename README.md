# mbank
`mbank` is a code for fast Gravitational Waves bank generation. It is very handy for generating precessing and eccentric banks.
The repository is still work-in-progress!

## How it works
Write something...

## How to install

To install the latest released verion (no release has been done yet):

```Bash
pip install mbank
```

To install the code from source:

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
	--variable-format Mq_s1xyz_s2z \
	--grid-size 6,1,1,1,1,3 \
	--MM 0.95 \
	--psd L1-REFERENCE_ASD-1164556817-1187740818.dat --asd \
	--f-min 15 --f-max 1024 \
	--mtot-range 15 75 \
	--q-range 2 5 \
	--s1-range 0.2 0.9 \
	--theta1-range 0 3.141592653 \
	--phi1-range 0 6.283185307 \
	--s2-range -0.9 0.9 \
	--plot \
	--placing-method geometric \
	--approximant IMRPhenomPv2 \
	--template-in-tile 40 \
	--use-ray 
```
To know more information about the available options type:
```Bash
mbank_run --help
```

You can also use the metric to estimate the fitting factor for a bunch of injections: 

```Bash
mbank_injections \
	--injs-per-tile 150\
	--N-neigh-templates 100 \
	--variable-format Mq_s1xyz_s2z \
	--psd L1-REFERENCE_ASD-1164556817-1187740818.dat --asd \
	--tiling-file out_test_bank/tiling_test_bank.npy \
	--bank-file out_test_bank/bank_test_bank.xml.gz \
	--approximant IMRPhenomPv2 \
	--f-min 15 --f-high 1024 \
	--plot \
	--full-match
```

If you specify the `--full-match` option, the match will be recomputed without a metric approximation: in this case, you want to speed things up with `--use-ray` and `--cache`.
You can also throw some injection chosen from a file: you just need to set an input xml injection file with the `--inj-file` option.

If you don't feel like typing all the options every time, you can add them to a text file and pass all of them through the `--ini-file` option. You can find an example ini file in the repo. To run it:

```Bash
mbank_run --ini-file test.ini
mbank_injections --ini-file test.ini
```

As you see, the same file can be used for different commands: each command will just ignore any option not relevant for it.


## Contacts

Fore more info, or just to say hello, you can contact me: [stefanoschmidt1995@gmail.com](mailto:stefanoschmidt1995@gmail.com)




















