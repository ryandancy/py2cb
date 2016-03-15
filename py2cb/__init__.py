#!usr/bin/env python
# -*- coding: utf-8

import argparse
import ast

__author__ = 'Copper'


class CommandBlock:

    IMPULSE = 0
    CHAIN = 1
    REPEAT = 2

    DOWN = 0
    UP = 1
    NORTH = 2
    SOUTH = 3
    WEST = 4
    EAST = 5

    CONDITIONAL = 8

    def __init__(self, command, type_=IMPULSE, metadata=EAST, auto=True):
        self.command = command
        self.type_ = type_
        self.metadata = metadata
        self.auto = auto

    def _get_command_block_name(self):
        return {
            CommandBlock.IMPULSE: 'command_block',
            CommandBlock.CHAIN: 'chain_command_block',
            CommandBlock.REPEAT: 'repeating_command_block'
        }[self.type_]

    def get_gen_command(self, offx, offy, offz):
        return 'setblock ~{0} ~{1} ~{2} minecraft:{3} {4} replace {{"Command":"{5}","auto":{6}b}}'.format(
            offx, offy, offz, self._get_command_block_name(), self.metadata, self.command, int(self.auto)
        )


def get_ast(code, filename):
    return ast.parse(code, filename=filename)


def parse_args():
    parser = argparse.ArgumentParser(description='Compiles Python to Minecraft command blocks')
    parser.add_argument('input_file', dest='infile', type=str, help='The Python file to be read.')

    output_group = parser.add_mutually_exclusive_group(required=True)
    output_group.add_argument(
        ['--output-file', '--output', '-o'], dest='dumpfile', type=str,
        help='The file to which commands will be dumped. Incompatible with --schematic-file/--schematic/-s')
    output_group.add_argument(
        ['--schematic-file', '--schematic', '-s'], dest='schemfile', type=str,
        help='The file to which the schematic will be dumped. Incompatible with --output-file/--output/-o')
    return parser.parse_args()
