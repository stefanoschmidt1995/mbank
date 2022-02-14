How to generate a bank
======================

To generate a bank, you need to specify the following options to the command `mbank_run`:

- `run-name`: a label for the run. All the output files will be named accordingly.
- `variable-format`: the coordinates to include in the bank. See [here](../package_reference/handlers.rst) the available formats.
- `mm`: minimum match requirement for the bank. It sets the average distance between templates
- `run-dir`: run directory. All the output will be stored here. Unless stated otherwise, all the inputs is understood to be here.
- `psd`: a psd file. If the option `asd` is set, the is understood to keep an ASD. The `ifo` option controls the interferometer to read the PSD of
- `placing-method`: the placing method to be used
- `grid-size`: set the size of the first coarse division along each variable. If the `use-ray` option is set, each coarse division will run in parallel
- `template-in-tile`: maximum number of templates in each tile that are tolerated in the iterative splitting. The smaller, the more precise is the tiling
- `approximant`: the lal waveform approximant to use for the metric computation
- `f-min`, `f-max`: the start and end frequency for the match (and metric) computation
- `var-range`: sets the boundaries for the variable `var`.
The possible variables are: `mtot`, `q`, `s1`/`s2`, `theta` (polar angle of spins), `phi` (azimuthal angle of spins), `iota` (inclination), `ref-phase` (reference phase), `e` (eccentricity), `meanano` (mean periastron anomaly).
- `plot`: create the plots?
- `show`: show the plots?
- `use-ray`: whether to parallelize the metric computation using the [`ray` package](https://www.ray.io/)

The bank will be saved in folder `run-dir` under the name `bank_run_name`, both in `npy` and `xml` format. In addition a tiling file and a `tile_id_population` file will be produced. The latter keeps track of the indices of the templates in each tile.

## Executing the commands

If you want to generate your first precessing bank, the options can be packed into a nice [`my_first_precessing_bank.ini`](https://github.com/stefanoschmidt1995/mbank/tree/master/examples/my_first_precessing_bank.ini) file like this:

```ini
[my_first_precessing_bank]
variable-format: Mq_s1xz
mm: 0.97
run-dir: precessing_bank
psd: ./H1L1-REFERENCE_PSD-1164556817-1187740818.xml.gz
ifo: L1
asd: false
placing-method: geometric
grid-size: 1,1,2,2
template-in-tile: 100
#approximant: EccentricFD
approximant: IMRPhenomPv2
f-min: 10
f-max: 1024
mtot-range: 25 30
q-range: 1 5
s1-range: 0.0 0.9 
s2-range: -0.9 0.9 
theta-range: -0.0 3.15
e-range: 0. 0.5 
phi-range: -3.15 3.15 
iota-range: 0.0 3.15
plot: true
show: true
use-ray: true
```

The `[section]` specification is compulsory: this will set the `run-name` variable!
You can then create your first precessing bank by

	mbank_run my_first_precessing_bank.ini

If the `--plot` option is set, you will see in your `--run-dir` some plot describing your bank:

![](../img/tiling.png)

![](../img/hist.png)

If you are happy with your tiling but you want to run again the template placing, you can change the ini file accordingly. Do not forget to specify the name of a tiling file!
For instance, if you want to change the minimum match, you can simply run:
	
	mbank_place_templates --tiling-file tiling_my_first_precessing_bank.npy --mm 0.95 my_first_precessing_bank.ini

This will do the placing again and will produce a new bank and new plots.

## Doing a bank by hand

WRITEME








