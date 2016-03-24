#!usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import argparse
from typing import List, Any

from py2cb import compiler

__author__ = 'Copper'


def pretty_print(array: List[List[Any]]) -> str:
    """array's items must all be equal in length"""
    max_lens = [0] * len(array[0])
    res = ''
    
    for row in array:
        for column, cell in enumerate(row):
            max_lens[column] = max(len(str(cell)), max_lens[column])
    
    max_lens = [x + 4 for x in max_lens]
    
    for row in array:
        for column, cell in enumerate(row):
            res += '{1:<{0}}'.format(max_lens[column], str(cell).replace('\n', r'\n'))
        res += '\n'
    
    return res


if sys.stdout.isatty():
    import colorama as colourama
    colourama.init(autoreset=True)

parser = argparse.ArgumentParser(description='Compiles Python to Minecraft command blocks')
parser.add_argument('input_file', type=str, help='The Python file to be read.')

output_group = parser.add_mutually_exclusive_group(required=True)
output_group.add_argument(
    '--output-file', '--output', '-o', type=str,
    help='The file to which commands will be dumped. Incompatible with --schematic-file/--schematic/-s')
output_group.add_argument(
    '--schematic-file', '--schematic', '-s', type=str,
    help='The file to which the schematic will be dumped. Incompatible with --output-file/--output/-o')

parsed_args = parser.parse_args()

contraption = compiler.parse(parsed_args.input_file)

if parsed_args.output_file:
    dump = contraption.get_dump()
    with open(parsed_args.output_file, 'w') as dumpfile:
        dumpfile.write(pretty_print(dump))
elif parsed_args.schematic_file:
    import pynbt
    with open(parsed_args.schematic_file, 'wb') as schemfile:
        contraption.get_schematic().save(schemfile, compression=pynbt.NBTFile.Compression.GZIP)
else:
    # This shouldn't happen, dumpfile & schemfile are mutually exclusive, required args...
    raise Exception('Either a dump or schematic file must be specified.')
