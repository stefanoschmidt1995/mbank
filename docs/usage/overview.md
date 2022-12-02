Overview
========

`mbank` offers an easy-to-use interface to generate and validate a template bank of gravitational waves signals. It makes use, whenever it makes sense, of a metric approximation, to speed up the computation of the match between two waveforms.

It is useful to think of a template banks as a Riemaniann _manifold_ equipped with a distance between points. The coordinates of the manifold are the components of the binary system (masses and spins) and possibly the orientation angles and the eccentric parameters: this makes a very large 12 dimensional manifold. Depending on the types of signals to detect, one can choose different sub-manifolds (lower dimensional), parameterized by a different set of variables. A template bank is a set of discrete points that covers as evenly as possible the manifold: it can be thought as a grid according to the complicated distance between points.
You can learn more on gravitational wave templates bank [here](https://journals.aps.org/prd/abstract/10.1103/PhysRevD.77.104017).

## Functionalities

The package implements four core functionalities:

1. **Tiling generation**: it partitions the parameter space into a set of non-overlapping tiles. A tile is an hyper-cube in the coordinates variables being used. In each tile, the match between two waveforms (geometrically, the distance between two points) is represented by a matrix, which approximates the match (distance) function with a bilinear form. The tiles are generated with an hierarchical splitting procedure, where each tile is split into sub-tiles if the variation of the metric determinant inside it is larger than a threshold.
The tiling can be supplemented with a normalizing flow model, which is able to interpolate within each tile, increasing the accuracy.

2. **Template placing**: given a tiling, it places templates so that they are (approximately) equally spaced in each tile. The distance between templates is controlled by the minimum match parameter. Several strategies are adopted:

	- _uniform_: a constant number of templates the templates are draw from an uniform distribution. There is no check for coverage. This works very poorly, as the resulting points are not well equally spaced.
	- _geometric_: the templates are placed on a grid according to the metric of each tile. This provides a very powerful coverage in the inner part of the tile but doesn't perform well at the boundaries between tiles.
	- _stochastic_: following a [standard technique](https://journals.aps.org/prd/abstract/10.1103/PhysRevD.80.104014), random templates are proposed and only the ones that are too far away from the others are included in the bank. This is the most accurate placing method that can be implemented, at the price of a large computational cost (but still smaller than other non-metric techniques).
	- _random_: taking inspiration from [here](https://arxiv.org/abs/2202.09380), random templates are added to the bank. The coverage is checked by means of a set of livepoints, i.e. a set of templates randomly drawn from the space. For each new template added, all the livepoints close to it are removed from the set of proposals. The iteration goes on until the set of livepoints is almost empty. This methods is fast and provides a good coverage, at the cost of placing more templates than needed. As the number of dimensions of the space increases, this method become more and more attractive.
	- _pruning_: This is the same method above where the proposals are drawn from the set of livepoints. While this make sure that the two added templates are never too close from each other, it is very memory expensive and unfeasible for large banks.
	- _tile\_stochastic_/_tile\_random_: the stochastic/random placement method is performed in each tile separately. Although this results in a huge speed-up, the parameter will likely be over-covered by the bank.
	- _iterative_: each tile is divided into sub-tiles according to the same algorithm used to generate the tiles. The metric is not computed from scratch. The templates are the centers of each sub-tile, as soon as the volume of the sub-tile is small enough.
	
3. **Injections recovery study**: given a set of injections, it computes the match of each injection with the bank. At first order, it uses the match approximation: this gives a very fast approximate assesment of the bank performance. It can also use the full match to compute the match.

4. **Generation of an injection file**: it builds a file with random injections, either in the tiles either randomly chose among the templates. This feature is useful for development of pipeline for GW searches.

## General structure

`mbank` comes with 4 executables that implement the functionalities descibed above. They are:

- ``mbank_run``: it generates a bank, both creating the tiling, placing the templates and (possibly) training the normalizing flow model
- ``mbank_place_templates``: given a tiling, it place the templates, possibly after training the normalizing flow model
- ``mbank_injections``: it computes the injection recovery of the bank.
- ``mbank_injfile``: builds the injection file

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

As many options are in commond with the all the `mbank` commands, you're encouraged to use the same ini both for all the tasks that `mbank` can do. This will let you to keep everything in a single place, ensuring consistency between runs.
You can find some complete example ini files [here](https://github.com/stefanoschmidt1995/mbank/tree/master/examples).

In next pages of the documentation, you can find more details on the actual use of the code.







