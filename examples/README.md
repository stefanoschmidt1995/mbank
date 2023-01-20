# Examples

This folder keeps some example `ini` files that you can run: hopefully they will help you to understand how things work. You can also explore the [docs](https://mbank.readthedocs.io/en/latest/usage/overview.html), if you're interested in more details.

- `my_first_eccentric_bank.ini` and `my_first_eccentric_bank.ini` provide a very simple example to generate a precessing bank or an eccentric bank and to perform injections: the bank generation and validation takes only a few minutes and can easily run on your laptop.

- Files `eccentric_paper.ini`, `HM_paper.ini` and `precessing_paper.ini` have been used to produce the case studies banks introduced in the paper: reproducing the results will most likely require you to use a high performance computer.

## Get things done

First things first, download your PSD:

```Bash
wget https://dcc.ligo.org/public/0165/T2000012/002/aligo_O3actual_H1.txt
```

To generate an eccentric bank:

```Bash
mbank_run my_first_eccentric_bank.ini
```

To validate with injections the bank generated you can type:

```Bash
mbank_injections my_first_eccentric_bank.ini
``` 

If you want to perform again the template placing (without generating the tiling)

``` Bash
mbank_place_templates my_first_eccentric_bank.ini
```

The same applies for the precessing bank (injections and bank generation options are kept in separate files).

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

