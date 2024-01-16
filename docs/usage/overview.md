Overview
========

`mbank` offers an easy-to-use interface to generate and validate a template bank of gravitational waves signals. It makes use, whenever it makes sense, of a metric approximation, to speed up the computation of the match between two waveforms.

It is useful to think of a template banks as a Riemaniann _manifold_ equipped with a distance between points. The coordinates of the manifold are the components of the binary system (masses and spins) and possibly the orientation angles and the eccentric parameters: this makes a very large 12 dimensional manifold. Depending on the types of signals to detect, one can choose different sub-manifolds (lower dimensional), parameterized by a different set of variables. A template bank is a set of discrete points that covers as evenly as possible the manifold: it can be thought as a grid according to the complicated distance between points.
You can learn more on gravitational wave templates bank [here](https://journals.aps.org/prd/abstract/10.1103/PhysRevD.77.104017).

## Functionalities

Central to `mbank` is a [normalizing flow](https://arxiv.org/abs/1912.02762) model to (i) sample the templates and (ii) estimate the template density (i.e. the volume element) on a given region of the parameter space.
The package implements several _core_ functionalities to generate a dataset for the flow training, to fit the flow, to place templates and to validate the template bank with injections. Other functionalities include template banks handling, volume estimation and comparison as well as some metric validation tools.

Below, we give you a hint of what `mbank` can do for you:

1. **Dataset generation**: it computes a dataset which is typically used to train the normalizing flow model. A daset consist in a set of points, each with the corresponding (log) volume element.

2. **Training of the normalizing flow**: it trains the normalizing flow model given a dataset.

3. **Template placing**: given a normalizing flow model (or a tiling, see below), it places templates so that they are (approximately) equally spaced. The distance between templates is controlled by the minimum match parameter.

	Taking inspiration from [here](https://arxiv.org/abs/2202.09380), random templates are added to the bank. The coverage is checked by means of a set of livepoints, i.e. a set of templates randomly drawn from the space. For each new template added, all the livepoints close to it are removed from the set of proposals. The iteration goes on until the set of livepoints is almost empty. This methods is fast and provides a good coverage, at the cost of placing more templates than needed. As the number of dimensions of the space increases, this method become more and more attractive.
	
	Other template placement strategies have been implemented, although not fully validated:

	- _uniform_: a constant number of templates the templates are draw from an uniform distribution. There is no check for coverage. This works very poorly, as the resulting points are not well equally spaced.
	- _geometric_: the templates are placed on a grid according to the metric of each tile. This provides a very powerful coverage in the inner part of the tile but doesn't perform well at the boundaries between tiles.
	- _stochastic_: following a [standard technique](https://journals.aps.org/prd/abstract/10.1103/PhysRevD.80.104014), random templates are proposed and only the ones that are too far away from the others are included in the bank. This is the most accurate placing method that can be implemented, at the price of a large computational cost (but still smaller than other non-metric techniques).
	
4. **Injections recovery study**: given a set of injections, it computes the match of each injection with the bank. At first order, it uses the match approximation: this gives a very fast approximate assesment of the bank performance. It can also use the full match to compute the match.

5. **Generation of an injection file**: it builds a file with random injections, either in the tiles either randomly chose among the templates. This feature is useful for development of pipeline for GW searches.

6. **Volume computation**: it provides a Monte Carlo estimate of the volume of the parameter space by using importance sampling. This is useful for forecasting the template bank size, as the volume of the space directly traslates into the number of templates inside a template bank.

During the development, the packaged changed a lot. The first version relied on the tiling to provide an estimation of the metric on the parameter space. A tiling partitions the parameter space into a set of non-overlapping an hyper-cubes in the coordinates variables being used. In each tile, the match between two waveforms (geometrically, the distance between two points) is represented by a matrix, which approximates the match (distance) function with a bilinear form.
The tiling was replaced by a normalizing flow model. The package still supports the old tiling but it is unadvised to use it. For more information on how to generate a template bank using the tiling, see the [old documentation page](bank_generation_tiling.md).

## General structure

`mbank` comes with several executables that implement the functionalities descibed above. They are:

- ``mbank_generate_flow_dataset``: it generates the dataset to train the normalizing flow model
- ``mbank_train_flow``: it trains the normalizing flow model
- ``mbank_place_templates``: given a trained flow, it place the templates
- ``mbank_injections``: it computes the injection recovery of the bank
- ``mbank_injections_workflow``: prepares a condor DAG to parallelize the injection jobs
- ``mbank_injfile``: builds the injection file
- ``mbank_print_metric``: prints the value of the metric at a given point, together with some additional information
- ``mbank_validate_metric``: produced some plots to validate the accuracy and the validity of the metric

Each executable comes with a number of options that are processed by a parser. For instance:

```Bash
mbank_run --options-you-like
```

If you don't feel too much like typing, you can specify some parameters on a `ini` file: the entries of the ini should have the same name of the relevant options. However, you can still use the parser, which will overwrite any conflicting option in the ini file. For example:

```Bash
mbank_run --other-options options_file.ini
```

If you're not sure which options are available, feel free to use the `--help` option.

```Bash
mbank_place_templates --help
```

As many options are in common with the all the `mbank` commands, you're encouraged to use the same ini both for all the tasks that `mbank` can do. This will let you to keep everything in a single place, ensuring consistency between runs.
You can find some complete example ini files [here](https://github.com/stefanoschmidt1995/mbank/tree/master/examples).

In next pages of the documentation, you can find more details on the actual use of the code.







