# Py2CB

A compiler that converts Python scripts to Minecraft command block sequences
via MCEdit schematics. Tested with Minecraft 1.8 and MCEdit 1.2.5.0.

Python scripts to be compiled use a limited subset of Python (ex: `range()` is
not supported) and interface with Minecraft through the `py2cb` library scripts
must import.

To use, run
`python <script file> <--schematic|--output> <schematic/output file>`.
`--output` causes the script to dump command block
data to the specified file in the format of a table (ignore the excessive
spaces at the end of each line). `--schematic` causes the script to output an
MCEdit schematic to the specified file, which can then be placed in a Minecraft
world by MCEdit and run. For example,
`python examples/roman_numerals.py --schematic roman-schem.schematic` causes
the Roman numeral-generating example script to be compiled into a schematic at
`roman-schem.schematic`.

Examples are found in `examples/`. For example, the `roman_numerals.py` script
generates a command-block chain which outputs sequential Roman numerals.

Depends on [`pynbt`](https://github.com/TkTech/PyNBT).