# Examples

This folder keeps some example `ini` files that you can run: hopefully they will help you to understand how things work.
You can also explore the [docs](https://mbank.readthedocs.io/en/latest/usage/overview.html), if you're interested in more details.

Below is what you will find in this folder:

- [`my_first_eccentric_bank.ini`](my_first_eccentric_bank.ini) provides a very simple example to generate a precessing bank or an eccentric bank and to perform injections: the bank generation and validation takes only a few minutes and can easily run on your laptop. The example is discussed in more details in the docs.

- Files [`HM_spinning_bank.ini`](HM_spinning_bank.ini) and [`precessing_bank.ini`](precessing_bank.ini) have been used to produce 
two large bank introduced in the [paper](https://arxiv.org/abs/2302.00436) (Sec. IV): reproducing the results will most likely requires you to use a high performance computer.

- [`validation.ini`](validation.ini): provides some options to make some validation plots of the metric (see [here](https://mbank.readthedocs.io/en/latest/usage/metric.html#validating-the-metric) for more details).

## Get things done

First things first, download your PSD:

```Bash
wget https://dcc.ligo.org/public/0165/T2000012/002/aligo_O3actual_H1.txt
```

To generate a dataset for the normalizing flow and to train it:

```Bash
mbank_generate_flow_dataset my_first_eccentric_bank.ini
mbank_train_flow my_first_eccentric_bank.ini
```

To generate an eccentric bank:

```Bash
mbank_place_templates my_first_eccentric_bank.ini
```

To generate injections and use them to validate with injections the bank generated you can type:

```Bash
mbank_injfile my_first_eccentric_bank.ini
mbank_injections my_first_eccentric_bank.ini
``` 

You can take a look at the plots produced by running the commands above in the [run folder](eccentric_bank).

To know more about the executables, you can check the [documentation](https://mbank.readthedocs.io/en/latest/usage/overview.html) or get help with:

```Bash
mbank_place_templates --help
```

and similarly for other commands.

You can also use the file `validation.ini` as an example for running `mbank_validate_metric`. It can give useful insight on the performance of the metric approximation:

```Bash
mbank_validate_metric validation.ini
```

See the [docs](https://mbank.readthedocs.io/en/latest/usage/metric.html#validating-the-metric) for more details on metric validation.

