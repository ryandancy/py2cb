#!usr/bin/env python
# -*- coding: utf-8

import argparse

__author__ = 'Copper'

parser = argparse.ArgumentParser(description='Compiles Python to Minecraft command blocks')
parser.add_argument('input_file', dest='infile', type=str, help='The Python file to be read.')
parser.add_argument(['--output-file', '--output', '-o'], dest='dumpfile', type=str,
                    help='The file to which commands will be dumped. Incompatible with --schematic-file/--schematic/-s')
parser.add_argument(['--schematic-file', '--schematic', '-s'], dest='schemfile', type=str,
                    help='The file to which the schematic will be dumped. Incompatible with --output-file/--output/-o')
parsed_args = parser.parse_args()
