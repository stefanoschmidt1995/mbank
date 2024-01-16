How to generate a bank
======================

The bank generation relies on three steps:

1. Dataset generation
2. Normalizing flow training 
3. Template placement

For the old version, which relies on the *tiling* for density estimation, you can refer to [this page](bank_generation_tiling.md). The page might be outdated and the template bank generated in this way will most likely be far from optimal.

To perform the three steps above there are three executables ``mbank_generate_flow_dataset``, ``mbank_train_flow`` and ``mbank_place_templates``. Clearly, they take several options to control their behaviour. Some options are in common, while others are specific to the each command.

To know which options are available for each command, you can use the command line to run the command you're interested in with the `--help` option; for instance:

```Bash
mbank_place_templates --help
```

The output of the help of all the available commands is also accessible at [this](../package_reference/bin.rst) page.








