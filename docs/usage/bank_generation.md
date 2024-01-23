How to generate a bank
======================

The bank generation happens in three logical steps:

1. Dataset generation
2. Normalizing flow training 
3. Template placement

For the old version, which relies on the *tiling* for density estimation, you can refer to [this page](bank_generation_tiling.md). The page might be outdated and the template bank generated in this way will most likely be far from optimal.

## From command line

The good news is that if you want to generate a template bank, you don't have to write any new piece of code.
Indeed, to perform the three steps above there are three executables:

- ``mbank_generate_flow_dataset``: to generate a dataset for the normalizing flow dataset
- ``mbank_train_flow``: to use the previously generated dataset to train a normalizing flow model
- ``mbank_place_templates``: to use the normalizing flow to place template according to a given minimal match

Clearly, each take several options to control their behaviour. Some options are in common, while others are specific to the each command.

To know which options are available for each command, you can use the command line to run the command you're interested in with the `--help` option; for instance:

```Bash
mbank_place_templates --help
```

The output of the help of all the available commands is also accessible at [this](../package_reference/bin.rst) page.

### Gathering options on a `ini` file

It is convenient to gather all the relevant parameter in a single `ini` file. In this way, you type all your options in a single place and you make sure that different executables have access to the same options.

If you are dreaming of generating a template bank of [eccentric](https://en.wikipedia.org/wiki/Orbital_eccentricity) BBH signals, a `ini` file could look like this.

```ini
[my_first_eccentric_bank]

	#File settings
run-dir: eccentric_bank
dataset: dataset_eccentric.dat
flow-file: flow_eccentric.zip
bank-file: my_first_eccentric_bank.dat

	#Metric settings
variable-format: Mq_nonspinning_e
psd: ./aligo_O3actual_H1.txt
asd: true
metric-type: symphony
approximant: EccentricFD
f-min: 10
f-max: 1024
df: 0.5

	#Parameter space ranges
mtot-range: 25 50
q-range: 1 5
e-range: 0. 0.4

	#Dataset generation & flow training options
n-datapoints: 3000
n-layers: 2
hidden-features: 30
n-epochs: 100000
learning-rate: 0.005
patience: 20
min-delta: 1e-3
batch-size: 500
train-fraction: 0.8
load-flow: false
ignore-boundaries: false
only-ll: true

	#Placing method options
placing-method: random
n-livepoints: 1000
covering-fraction: 0.9
mm: 0.97

	#Injection generation options
gps-start: 1239641219
gps-end: 1239642219
time-step: 10
inj-out-file: eccentric_injections.xml

	#Injection options
n-injs: 500
inj-file: eccentric_injections.xml
full-symphony-match: true
metric-match: false

	#Communication with the user
plot: true
show: true
verbose: true
```

These are a lot of parameters. Without being exhaustive, we describe below the most important of them:

- `variable-format`: the coordinates to include in the bank. See [here](variable_handler) the available formats.
- `run-dir`: run directory. All the output will be stored here. Unless stated otherwise, all the inputs is understood to be located in this folder.
- `psd`: a psd file. If the option `asd` is set, the is understood to keep an ASD. The `ifo` option controls the interferometer to read the PSD of
- `mm`: minimum match requirement for the bank. It sets the average distance between templates
- `metric-type`: the metric computation algorithm to use. It is advised to use `symphony` for precessing and/or HM systems, while `hessian` for the others. Other options are possible, without being tested extensively.
- `placing-method`: the placing method to be used. While many are available, only the `random` method has been extensively tested, as discussed in the publication.
- `livepoints`: the number of livepoints to be used for the random placing methods. This is the number of points that cover the space initially. They will be removed as soon as the bank grows in size.
- `covering-fraction`: the fraction of the space to be covered before stopping the template bank generation. The covering fraction is computed with a monte carlo estimation using the livepoints.
- `learning-rate`: learning rate for the training loop
- `min-delta` and `patience`: parameters to control the early stopping
- `approximant`: the lal waveform approximant to use for the metric computation
- `f-min`, `f-max`: the start and end frequency for the match (and metric) computation
- `var-range`: sets the boundaries for the variable `var`.
The possible variables are: `mtot`, `q`, `s1`/`s2`, `theta` (polar angle of spins), `phi` (azimuthal angle of spins), `iota` (inclination), `ref-phase` (reference phase), `e` (eccentricity), `meanano` (mean periastron anomaly).
- `n-layers`: number of layers to be used in the flow architecture. Each layer is formed by a Linear layer + a Masked Affine Autoregressive layer
- `hidden-features`: number of hidden features in each Masked Affine Autoregressive layer
- `n-epochs`: the number of training epochs for the flow
- `plot`: create the plots?
- `show`: show the plots?

Besides those parameters, we included a few parameters which are relevant for an injection study. Please, read more in the [next section](injections.md).

You can easily run by yourselfs all the commands below and in ten minutes, you will have a nice template bank.

```bash
mbank_generate_flow_dataset my_first_eccentric_bank.ini
mbank_train_flow my_first_eccentric_bank.ini
mbank_place_templates my_first_eccentric_bank.ini
```

They will produce a lots of plots, to validate your normalizing flow performance and to show the template distribution within the bank.
This is the accuracy of the normalizing flow to reproduce the true density {math}`\sqrt{M(\theta)}` of the parameter space with its PDF {math}`p_\text{flow}(\theta)`.

![](../img/flow_accuracy.png)

Finally, you can also see the resulting template bank.

![](../img/bank_eccentric.png)

You can find the [ini file](https://github.com/stefanoschmidt1995/mbank/blob/master/examples/my_first_eccentric_bank.ini) and all the plots produced by the code in the [example folder](https://github.com/stefanoschmidt1995/mbank/tree/master/examples) of the repository.

## Bank by hands

Of course you can also code the bank generation by yourself in a python script. Althought this requires more work, it gives more control on the low level details and in some situation can be useful. However, for ease of use, it is always advised to use the provided executables `mbank_run` and `mbank_place_templates`.

WRITEME!!!





