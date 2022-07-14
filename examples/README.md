# Examples

This folder keeps some example `ini` files. They can be used to generate a precessing bank or an eccentric bank and to perform injections.

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

To know more about the commands, you can check the [documentation](link) or get help with:

```Bash
mbank_place_templates --help
```

or similarly for other commands.
