#!usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import ast

from pynbt import NBTFile, TAG_Byte_Array, TAG_Compound, TAG_List, TAG_Short, TAG_String
from typing import Tuple, List, Any


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
    
    def __init__(self, command: str, type_: int, metadata: int = EAST, auto: bool = True) -> None:
        self.command = command
        self.type_ = type_
        self.metadata = metadata
        self.auto = auto
    
    def get_command_block_name(self) -> str:
        return {
            CommandBlock.IMPULSE: 'command_block',
            CommandBlock.CHAIN: 'chain_command_block',
            CommandBlock.REPEAT: 'repeating_command_block'
        }[self.type_]
    
    def get_block_id(self) -> int:
        return {
            CommandBlock.IMPULSE: 137,
            CommandBlock.CHAIN: 211,
            CommandBlock.REPEAT: 210
        }[self.type_]
    
    def get_gen_command(self, offx: int, offy: int, offz: int) -> str:
        return 'setblock ~{0} ~{1} ~{2} minecraft:{3} {4} replace {{"Command":"{5}","auto":{6}b}}'.format(
            offx, offy, offz, self.get_command_block_name(), self.metadata, self.command, int(self.auto)
        )
    
    def get_dump(self) -> List[str]:
        """Generates a list of Type, Metadata, Auto, Commmand. Mostly exists for Contraption.get_dump."""
        return [self.get_command_block_name(), self.metadata, self.auto, self.command]


class Contraption:
    
    def __init__(self) -> None:
        self.cblocks = []  # self.cblocks: List[Tuple[Tuple[int, int, int], CommandBlock]]
    
    def add_block(self, xyz: Tuple[int, int, int], block: CommandBlock) -> None:
        self.cblocks.append((xyz, block))
    
    def get_dump(self) -> List[List[str]]:
        """Generates a table of command block X, Y, Z, Type, Metadata, Auto, Command."""
        header = ['X', 'Y', 'Z', 'Type', 'Metadata', 'Auto', 'Command']
        res = [header]
        for xyz, cblock in self.cblocks:
            res.append(list(xyz) + cblock.get_dump())
        return res
    
    def get_schematic(self) -> NBTFile:
        """Exports this contraption to a schematic NBT file."""
        # Uses unofficial .schematic format found at the Minecraft Wiki (minecraft.gamepedia.com/Schematic_file_format)
        nbt = NBTFile(name='')
        
        width = max(x for (x, y, z), cblock in self.cblocks) + 1
        height = max(y for (x, y, z), cblock in self.cblocks) + 1
        length = max(z for (x, y, z), cblock in self.cblocks) + 1
        
        # blocks and data are sorted by height/y, then length/z, then width/x (YZX)
        # therefore the index of x, y, z in blocks/data is (y * length + z) * width + x
        blocks = data = [0] * (width * length * height)  # 0 is air
        for (x, y, z), cblock in self.cblocks:
            index = (y * length + z) * width + x
            blocks[index] = cblock.get_block_id()
            data[index] = cblock.metadata
        
        nbt['Width'] = TAG_Short(width)
        nbt['Height'] = TAG_Short(height)
        nbt['Length'] = TAG_Short(length)
        nbt['Materials'] = TAG_String('Alpha')
        nbt['Entities'] = TAG_List(TAG_Compound, [])
        nbt['TileEntities'] = TAG_List(TAG_Compound, [])
        nbt['Blocks'] = TAG_Byte_Array(blocks)
        nbt['Data'] = TAG_Byte_Array(data)
        
        return nbt


class IDContainer:
    
    _vars_to_ids = {}
    _id_counter = 0
    
    def _next_id(self) -> int:
        if self._id_counter >= 0:
            self._id_counter += 1
            if self._id_counter == 2 ** 31:
                self._id_counter = -1
        else:
            self._id_counter -= 1
            if self._id_counter < -(2 ** 31):
                raise Exception('IDContainer ran out of IDs!')
        return self._id_counter
    
    def add(self, var: Any) -> None:
        # Silently ignores adding multiple times
        if var not in self._vars_to_ids:
            self._vars_to_ids[var] = self._next_id()
    
    def get_id(self, var: Any) -> int:
        return self._vars_to_ids[var]
    
    def has_var(self, var: Any) -> bool:
        return var in self._vars_to_ids


stringids = IDContainer()
exprids = IDContainer()


consts = []


def add_const(const: int, contr: Contraption, x: int, y: int, z: int) -> Tuple[Contraption, int, int, int]:
    if const not in consts:
        x += 1
        contr.add_block((x, y, z), CommandBlock('scoreboard players set const_{0} py2cb_intrnl {0}'.format(const),
                                                CommandBlock.CHAIN))
        consts.append(const)
    return contr, x, y, z


def parse_node(node: ast.AST, contr: Contraption, x: int, y: int, z: int) -> Tuple[Contraption, int, int, int]:
    # ASSIGNMENTS
    if isinstance(node, ast.Assign):
        for target in node.targets:
            # Assignment with names (n = _)
            if isinstance(target, ast.Name):
                # Simple assignment - name = num (ex: n = 4)
                if isinstance(node.value, ast.Num):
                    x += 1
                    contr.add_block((x, y, z), CommandBlock('scoreboard players set {0} py2cb_var {1}'
                                                            .format(target.id, node.value.n), CommandBlock.CHAIN))
                
                # Simple assignment - name = str (ex: n = 'foo')
                elif isinstance(node.value, ast.Str):
                    # Strings are represented by armor stands with custom names
                    x += 1
                    contr.add_block((x, y, z), CommandBlock(
                        'summon ArmorStand ~ ~1 ~ {{"NoGravity":1b,"CustomName":{0},"Tags":["string_noname"]}}'
                        .format(node.value.s), CommandBlock.CHAIN))
                    
                    x += 1
                    stringids.add(target.id)
                    contr.add_block((x, y, z), CommandBlock(
                        'scoreboard players set @e[type=ArmorStand,tag=string_noname] py2cb_var {0}'
                        .format(stringids.get_id(target.id)), CommandBlock.CHAIN))
                    
                    x += 1
                    contr.add_block((x, y, z), CommandBlock(
                        'entitydata @e[type=ArmorStand,tag=string_noname] {"Tags":["string"]}', CommandBlock.CHAIN))
                
                # Simple assignment - name = True/False/None (ex: n = True)
                elif isinstance(node.value, ast.NameConstant):
                    x += 1
                    contr.add_block((x, y, z), CommandBlock(
                        'scoreboard players set {0} py2cb_var {1}'
                        .format(target.id, int(node.value.value)), CommandBlock.CHAIN))
                
                # Not-so-simple assignment - name = expr (ex: n = 39 + 2 * 3)
                elif isinstance(node.value, ast.Expr):
                    contr, x, y, z = parse_node(node.value, contr, x, y, z)
                    x += 1
                    contr.add_block((x, y, z), CommandBlock(
                        'scoreboard players operation {0} py2cb_var = expr_{1} py2cb_intrnl'
                        .format(target.id, exprids.get_id(node.value)), CommandBlock.CHAIN))
    
    # BINOPS
    elif isinstance(node, ast.BinOp):
        # Addition
        if isinstance(node.op, ast.Add):
            for side in [node.left, node.right]:
                if isinstance(side, ast.Num):
                    contr, x, y, z = add_const(side.n, contr, x, y, z)
            
            # <= is issubset operator on sets
            if set(map(type, [node.left, node.right])) <= {ast.Num, ast.Name}:
                x += 1
                exprids.add(node)
                contr.add_block((x, y, z), CommandBlock(
                    'scoreboard players operation expr_{0} py2cb_intrnl = {1}'.format(
                        exprids.get_id(node),
                        'const_{0} py2cb_intrnl'.format(node.left.n)
                        if isinstance(node.left, ast.Num) else
                        '{0} py2cb_var'.format(node.left.id)
                    ),
                    CommandBlock.CHAIN
                ))
                x += 1
                contr.add_block((x, y, z), CommandBlock(
                    'scoreboard players operation expr_{0} py2cb_intrnl += {1}'.format(
                        exprids.get_id(node),
                        'const_{0} py2cb_intrnl'.format(node.right.n)
                        if isinstance(node.right, ast.Num) else
                        '{0} py2cb_var'.format(node.right.id)
                    ),
                    CommandBlock.CHAIN
                ))
    
    return contr, x, y, z


def parse(ast_root: ast.Module) -> Contraption:
    res = Contraption()
    x = y = z = 0
    res.add_block((x, y, z), CommandBlock('scoreboard objectives add py2cb_intrnl dummy Py2CB Internal Variables',
                                          CommandBlock.IMPULSE, auto=False))
    x += 1
    res.add_block((x, y, z), CommandBlock('scoreboard objectives add py2cb_var dummy Py2CB Application Variables',
                                          CommandBlock.CHAIN))
    
    for statement in ast_root.body:
        res, x, y, z = parse_node(statement, res, x, y, z)
    return res


def get_ast(code: str, filename: str) -> ast.AST:
    return ast.parse(code, filename=filename)


def parse_args() -> argparse.Namespace:
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
