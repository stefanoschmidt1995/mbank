How to perform injections
=========================

The injection code throws injections in the bank and computes the match of each of them with the templates. The injections can be either loaded by file either randomly generated within each tile.

Many options are in common with `mbank_run` and `mbank_place_templates`. The options unique to the `mbank_injections` are:

- `tiling-file`: name of the tiling file
- `bank-file`: name of the bank file
- `inj-file`: an xml file to load a generic set of injection from. If it's not set, the injections will be drawn inside the tiling.
- `n-injs`: how many injections to perform? They will randomly placed in the space so that each tile will keep a number of injections proportional to the volume.
- `seed`: random seed for the injections
- `full-match`: whether to compute the full match, rather than just the metric approximation
- `N-neigh-templates`: number of neighbours template (in euclidean distance) to compute the metric match with
- `N-neigh-templates-full`: number of neighbours template (in metric distance) to compute the full match with.


## Injections from command line
Assuming you generated the bank normally, here's how an injection file [`my_first_precessing_injections.ini`](https://github.com/stefanoschmidt1995/mbank/tree/master/examples/my_first_precessing_injections.ini) could look like (but you're suggested to keep all the options in a single file):

```ini
[my_first_precessing_bank]

variable-format: Mq_s1xz
mm: 0.97
run-dir: precessing_bank
psd: ./aligo_O3actual_H1.txt
ifo: H1
asd: true
approximant: IMRPhenomXP
f-min: 15
f-max: 1024
plot: true
show: true
use-ray: true

tiling-file: tiling_my_first_precessing_bank.npy
bank-file: bank_my_first_precessing_bank.xml.gz

n-injs: 10000
seed: 0
N-neigh-templates: 1000
N-neigh-templates-full: 80 
full-match: false
```

By running

	mbank_injections my_first_precessing_injections.ini

This will produce two nice plots.

The histogram of the fitting factor of each injection (i.e. best match of an injection with the templates)

![](../img/FF_hist.png)

and a scatter plot with the injections with fitting factor smaller that `mm`: 
![](../img/injections.png)

## Injections by hands

Again, we can also perform injections using a python script (although this is not advised).
Here we assume we have at hand a three dimensional bank `bank.dat` and a tiling `tiling.npy`, with the variable format `Mq_chi`: this was generated in the previous [page](../usage/bank_generation.md).

After the imports,

```Python
from mbank import variable_handler, cbc_metric, cbc_bank, tiling_handler
from mbank.utils import compute_injections_match, compute_metric_injections_match
from mbank.utils import load_PSD, plot_tiles_templates
import numpy as np
```

you need to load the bank, the tiling and to instantiate a variable handler:

```Python
bank = cbc_bank('Mq_chi', 'bank.dat')
t_obj = tiling_handler('tiling.npy')
var_handler = variable_handler()
```
We then generate the injection sampling them from the tiling and compute the match with the bank:

```Python
n_injs = 1000
injs_3D = t_obj.sample_from_tiling(n_injs)
stat_dict = compute_metric_injections_match(injs_3D, bank, t_obj, N_neigh_templates = 1000)
```
The function will return a dictionary with the injections statistics computed: note that since we are using the metric approximation to the match, it runs very fast. To know more about the entries of the dictionary, you can take a look at the documentation of `mbank.utils.compute_metric_injections_match`.
The output dictionary also keeps the value of the injections in the full 12 dimensional BBH space, so that you don't need to worry to save them separately.

If you want to compute the full match, you can do so, after the computation of the _metric_ match, with the function `compute_injections_match`. Make sure to define a metric object first and transform your templates in the full 12 dimensional BBH space!

```Python
metric = cbc_metric(bank.variable_format,
			PSD = load_PSD('aligo_O3actual_H1.txt', True, 'H1'),
			approx = 'IMRPhenomD',
			f_min = 10, f_max = 1024)
templates_full = np.array(var_handler.get_BBH_components(bank.templates, bank.variable_format)).T
stat_dict = compute_injections_match(stat_dict, templates_full, metric,
			N_neigh_templates = 100, symphony_match = False, cache = False)
```
This will update some of the entries of the `stat_dict` with the unapproximated fitting factors: as the generation of many WFs is required, this will take a while.

We can then plot the injections performed on the tiling and we colour them by their fitting factor:

```Python
best_matches = stat_dict['match'] if stat_dict['match'] is not None else stat_dict['metric_match']
plot_tiles_templates(t_obj, bank.templates, bank.variable_format, var_handler,
			injections = injs_3D, show = True,
			inj_cmap =  best_matches)
```







