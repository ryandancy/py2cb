# Py2CB

A compiler that converts Python scripts to Minecraft command block sequences
via MCEdit schematics. Tested with Minecraft 1.8 and MCEdit 1.2.5.0.

Python scripts to be compiled use a limited subset of Python (ex: `range()` is
not supported) and interface with Minecraft through functions found in
`py2cb/script.py` and imported through `import py2cb`.

## Usage

To use, run
`python py2cb <script file> <--schematic|--output> <schematic/output file>`.

* `--output` causes the script to dump command block
data to the specified file in the format of a table (ignore the excessive
spaces at the end of each line).
* `--schematic` causes the script to output an
MCEdit schematic to the specified file, which can then be placed in a Minecraft
world by MCEdit and run.

For example,
`python examples/roman_numerals.py --schematic roman-schem.schematic` causes
the Roman numeral-generating example script to be compiled into a schematic at
`roman-schem.schematic`.

Alternatively, you can run the script file itself to test before importing it
into Minecraft. The functions found in `py2cb/script.py` contain sensible
defaults which attempt to mimic the Minecraft output in-terminal.

## Example scripts

Examples are found in `examples/`. For example, the `roman_numerals.py` script
generates a command-block chain which outputs sequential Roman numerals.

## Supported Python language constructs

* Variable assignment:
    - to numbers: `n = 5`, `m = -1.52`. Numerical variables (and constants) are
      stored as scoreboard variables.
    - to strings: `s = 'hello, world!'`. String variables are stored as entity
      (armour stand) names. Note that because of this, the empty string cannot
      be assigned to - entities cannot be named the empty string.
    - to `True`, or `False`: `b = True`. Note that `None` is not supported.
    - to a list: `my_list = [1, 2, 3, 4]`. Lists are done using entities with
      special names.
    - to other variables: `a = b`.
    - to expressions: `n = 2 * 4 - 2`.
    - Multi-assignment is supported: `a = b = c`.
    - Note that variable names can be a maximum of 38 characters due to
      a Minecraft limit of 40 characters for scoreboard variables and Py2CB's
      reservation of the last 2 characters.
* Augmented assignment: `n += 5`.
* Binary, boolean, and unary operators: `a * 4`, `(b1 or b2) and b3`,
  `-x`.
* Comparison statements: `a >= b`, `n != 5`.
* Ternary 'if' expressions: `a if x else b`
* If statements
* While loops
* For loops with lists
* Lists:
    - assignment: `a = [1, 2, 3, 4]`.
    - indexing: `a[3]`, `b[n]`.
    - Note that strings aren't lists in py2cb scripts, and that indexing must
      be done on a variable (ex: `[1, 2, 3][n]` is unsupported).
* Functions:
    - `min()` and `max()`
    - `say()`, `tell()`, and `tellraw()`, as defined in `py2cb/script.py`. Used
      to interface with Minecraft chat and deliver messages to the player.
    - User-defined functions (definition and calling): `def foo(x): pass`,
      `foo('hello, world')`.

## Dependencies

Depends on [`pynbt`](https://github.com/TkTech/PyNBT).

Note: in the commit history, Copper is me from 2016.