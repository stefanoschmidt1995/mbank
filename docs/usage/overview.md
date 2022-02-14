Overview
========

`mbank` offers an easy-to-use interface to generate and validate a template bank of gravitational waves signals. It makes use, whenever it makes sense, of a metric approximation, to speed up the computation of the match between two waveforms.

It is useful to think of a template banks as a Riemaniann manifold equipped with a distance between points. The coordinates of the manifold are the components of the binary system (masses and spins) and possibly the orientation angles and the eccentric parameters: this makes a very large 12 dimensional manifold. Depending on the types of signals to detect, one can choose different sub-manifolds (lower dimensional), parameterized by a different set of variables. A template bank is a set of discrete points that covers as evenly as possible the manifold: it can be taught as a grid according to the complicated distance between points.
You can learn more on gravitational wave templates bank [here](where/is/this/link?).

## Functionalities

The package implements four core functionalities:

1. **Tiling generation**: it partition the parameter space into a set of non-overlapping tiles. A tile is an hyper-cube in the coordinates variables being used. In each tile, the match between two waveforms (geometrically, the distance between two points) is represented by a matrix, which approximates the match (distance) function to a bilinear form. The tiles are generated with an hierarchical splitting procedure: where each tile is split into sub-tiles if its volume accomodates for more templates than a given threshold.

2. **Template placing**: given a tiling, it place templates so that they are (approximately) equally spaced in each tile. The distance between templates is controlled by the minimum match parameter. Several tiling strategies are adopted:

	- _uniform_: all the templates are draw from an uniform distribution. This works very poorly, as the resulting points are not well equally spaced
	- _geometric_: the templates are placed on a grid according to the metric of each tile. This provides a very powerful coverage in the inner part of the tile but doesn't perform well at the boundaries between tiles.
	- _stochastic_: random templates are propose and only the ones that are too far away from the others are included in the bank. This method is used to refine the placement provided by the _geometric method_. This is the most accurate placing method that can be implemented. On the other hand, the stochastic search can be time consuming.
	- _pure\_stochastic_: the stochastic placement described above is used without any seed bank. This means that every template added is proposed at random. While very accurate, it can be very very slow.
	- _iterative_: each tile is divided into sub-tiles according to the same algorithm used to generate the tiles. The metric is not computed from scratch. The templates are the centers of each sub-tile, as soon as the volume of the sub-tile is small enough.
	
3. **Injections recovery study**: given a set of injections, it computes the match of each injection with the bank. At first order, it uses the match approximation: this gives a very fast approximate assesment of the bank performance. It can also use the full match to compute the match.

4. **Generation of an injection file**: it builds a file with random injections, either in the tiles either randomly chose among the templates. This feature is useful for development of pipeline for GW searches.

## General structure

`mbank` comes with 4 executables that implement the functionalities descibed above. They are:

- ``mbank_run``: it generates a bank, both creating the tiling and placing the templates
- ``mbank_place_templates``: given a tiling, it place the templates
- ``mbank_injections``: it computes the injection recovery of the bank.
- ``mbank_injfile``: builds the injection file

Each executable comes with a number of options that are processed by a parser. For instance:

	mbank_run --options-you-like

If you don't feel too much like typing, you can specify some parameters on a `ini` file: the entries of the ini should have the same name of the relevant options. However, you can still use the parser, which will overwrite any conflicting option in the ini file. For example:

	mbank_run --other-options options_file.ini

If you're not sure which options are available, feel free to use the `--help` option.

	mbank_place_templates --help

As many options are in commond with the all the `mbank` commands, you're encouraged to use the same ini both for all the tasks that `mbank` can do. This will let you to keep everything in a single place, ensuring consistency between runs.
You can find some complete example ini files [here](get/ini/file).

In next pages of the documentation, you can find more details on the actual use of the code.







