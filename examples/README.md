# Examples

This folder keeps some example `ini` files. They can be used to generate a precessing bank or an eccentric bank and to perform injections.

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

