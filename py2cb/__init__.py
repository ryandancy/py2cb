#!usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import ast

from pynbt import NBTFile, TAG_Byte, TAG_Byte_Array, TAG_Compound, TAG_Int, TAG_List, TAG_Short, TAG_String
from typing import Tuple, List, Any, Optional

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
    
    def __init__(self, command: str, type_: int = CHAIN, metadata: int = EAST, auto: bool = True) -> None:
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
        return 'setblock ~{0} ~{1} ~{2} minecraft:{3} {4} replace {{Command:"{5}",auto:{6}b}}'.format(
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
        """
        Exports this contraption to a schematic NBT file.
        Remember to pass compression=NBTFile.Compression.GZIP when calling save!
        """
        # Uses unofficial .schematic format found at the Minecraft Wiki (minecraft.gamepedia.com/Schematic_file_format)
        # (ish, because MCEdit uses a 'Biomes' TAG_Byte_Array not documented on the wiki)
        nbt = NBTFile(name='Schematic')
        
        width = max(x for (x, y, z), cblock in self.cblocks) + 1
        height = max(y for (x, y, z), cblock in self.cblocks) + 1
        length = max(z for (x, y, z), cblock in self.cblocks) + 1
        
        # blocks and data are sorted by height/y, then length/z, then width/x (YZX)
        # therefore the index of x, y, z in blocks/data is (y * length + z) * width + x
        blocks = [0] * (width * length * height)  # 0 is air
        data = [0] * (width * length * height)
        tiles = []
        for (x, y, z), cblock in self.cblocks:
            index = (y * length + z) * width + x
            blocks[index] = cblock.get_block_id()
            data[index] = cblock.metadata
            
            tiles.append(TAG_Compound(value={
                'x': TAG_Int(x),
                'y': TAG_Int(y),
                'z': TAG_Int(z),
                'Command': TAG_String(cblock.command),
                'auto': TAG_Byte(int(cblock.auto)),
                # everything below here is the same for every cblock
                'id': TAG_String('Control'),
                'powered': TAG_Byte(0),
                'conditionMet': TAG_Byte(0),
                'TrackOutput': TAG_Byte(1),
                'LastOutput': TAG_String(''),
                'CustomName': TAG_String('@')
            }))
        
        # Fudge the block ids to make them behave like unsigned bytes
        for i, blockid in enumerate(blocks):
            if blockid > 127:
                blocks[i] = blockid - 256
        
        nbt['Width'] = TAG_Short(width)
        nbt['Height'] = TAG_Short(height)
        nbt['Length'] = TAG_Short(length)
        nbt['Materials'] = TAG_String('Alpha')
        nbt['Entities'] = TAG_List(TAG_Compound, [])
        nbt['TileTicks'] = TAG_List(TAG_Compound, [])
        nbt['Blocks'] = TAG_Byte_Array(blocks)
        nbt['Data'] = TAG_Byte_Array(data)
        nbt['TileEntities'] = TAG_List(TAG_Compound, tiles)
        nbt['Biomes'] = TAG_Byte_Array([127] * (width * length))  # The 'void' biome
        
        return nbt


class IDContainer:
    
    def __init__(self, has_limit=False):
        self.has_limit = has_limit
        self._vars_to_ids = {}
        self._id_counter = 0
    
    def _next_id(self) -> int:
        if self.has_limit:
            if self._id_counter >= 0:
                self._id_counter += 1
                if self._id_counter == 2 ** 31:
                    self._id_counter = -1
            else:
                self._id_counter -= 1
                if self._id_counter < -(2 ** 31):
                    raise Exception('IDContainer ran out of IDs!')
        else:
            self._id_counter += 1
        return self._id_counter
    
    def add(self, var: Any) -> None:
        # Silently ignores adding multiple times
        if var not in self._vars_to_ids:
            self._vars_to_ids[var] = self._next_id()
    
    def __getitem__(self, var: Any) -> int:
        return self._vars_to_ids[var]
    
    def __contains__(self, var: Any) -> bool:
        return var in self._vars_to_ids


stringids = IDContainer(has_limit=True)
exprids = IDContainer()
listids = IDContainer()

consts = []
num_branches = 1


def add_const(const: int, contr: Contraption, x: int, y: int, z: int) -> Tuple[Contraption, int, int, int]:
    if const not in consts:
        x += 1
        contr.add_block((x, y, z), CommandBlock('scoreboard players set const_{0} py2cb_intrnl {0}'.format(const)))
        consts.append(const)
    return contr, x, y, z


def get_player_and_obj(node: ast.AST) -> str:
    """Assumes that all relavant things are in place (const_n, etc)"""
    if isinstance(node, ast.Num):
        return 'const_{0} py2cb_intrnl'.format(node.n)
    elif isinstance(node, ast.Name):
        return '{0} py2cb_var'.format(node.id)
    elif isinstance(node, ast.NameConstant):
        return 'const_{0} py2cb_intrnl'.format(nameconstant_to_int(node))
    elif node in exprids:
        return 'expr_{0} py2cb_intrnl'.format(exprids[node])


def nameconstant_to_int(node: ast.NameConstant) -> int:
    if node.value is None:
        raise Exception('None may not be used in Py2CB files.')
    else:
        return int(node.value)


def get_op_char(op: ast.operator) -> str:
    try:
        return {
            ast.Add: '+',
            ast.Sub: '-',
            ast.Mult: '*',
            ast.FloorDiv: '/',
            ast.Mod: '%'
        }[type(op)]
    except KeyError:
        raise Exception('Invalid binary operation (only +, -, *, //, % are allowed).')


def setup_internal_values(node: ast.AST, contr: Contraption, x: int, y: int, z: int, redef: bool = False) \
        -> Tuple[Contraption, int, int, int]:
    if isinstance(node, ast.Num):
        contr, x, y, z = add_const(node.n, contr, x, y, z)
    elif isinstance(node, ast.NameConstant):
        contr, x, y, z = add_const(nameconstant_to_int(node), contr, x, y, z)
    elif not isinstance(node, ast.Name) and (True if redef else node in exprids):
        contr, x, y, z = parse_node(node, contr, x, y, z)
        if node not in exprids:
            exprids.add(node)
    return contr, x, y, z


def add_pulsegiver_block(contr: Contraption, x: int, y: int, z: int,
                         wx: Optional[int] = None, wy: Optional[int] = None, wz: Optional[int] = None,
                         conditional: bool = True) \
        -> Tuple[Contraption, int, int, int]:
    """(wx, wy, wz) defaults to (-x - 1, 0, num_branches - z - 1)"""
    if wx is None:
        offx = -x - 1
    else:
        offx = wx - x
    if wy is None:
        offy = 0
    else:
        offy = wy - y
    if wz is None:
        offz = num_branches - z - 1
    else:
        offz = wz - z
    
    x += 1
    contr.add_block((x, y, z), CommandBlock(
        CommandBlock(
            'setblock ~ ~ ~ minecraft:air', type_=CommandBlock.IMPULSE, auto=True
        ).get_gen_command(offx, offy, offz),
        metadata=CommandBlock.EAST | (CommandBlock.CONDITIONAL if conditional else 0)
    ))
    
    return contr, x, y, z


def parse_node(node: ast.AST, contr: Contraption, x: int, y: int, z: int) -> Tuple[Contraption, int, int, int]:
    global num_branches
    # ASSIGNMENTS
    if isinstance(node, ast.Assign):
        for target in node.targets:
            # Assignment with names (n = _)
            if isinstance(target, ast.Name):
                # Simple assignment - name = num (ex: n = 4)
                if isinstance(node.value, ast.Num):
                    x += 1
                    contr.add_block((x, y, z), CommandBlock(
                        'scoreboard players set {0} py2cb_var {1}'.format(target.id, node.value.n)
                    ))
                
                # Simple assignment - name = str (ex: n = 'foo')
                elif isinstance(node.value, ast.Str):
                    # Strings are represented by armor stands with custom names
                    x += 1
                    contr.add_block((x, y, z), CommandBlock(
                        'summon ArmorStand ~ ~1 ~ {{NoGravity:1b,CustomName:"{0}",Tags:["string_noname"]}}'
                            .format(node.value.s)
                    ))
                    
                    x += 1
                    stringids.add(target)
                    contr.add_block((x, y, z), CommandBlock(
                        'scoreboard players set @e[type=ArmorStand,tag=string_noname] py2cb_var {0}'
                            .format(stringids[target])
                    ))
                    
                    x += 1
                    contr.add_block((x, y, z), CommandBlock(
                        'entitydata @e[type=ArmorStand,tag=string_noname] {Tags:["string"]}'
                    ))
                
                # Simple assignment - name = True/False/None (ex: n = True)
                elif isinstance(node.value, ast.NameConstant):
                    x += 1
                    contr.add_block((x, y, z), CommandBlock(
                        'scoreboard players set {0} py2cb_var {1}'
                            .format(target.id, nameconstant_to_int(node.value))
                    ))
                
                # Simple assignment - name = list (ex: n = [5, 6, 7])
                # TODO Strings in lists
                elif isinstance(node.value, ast.List):
                    listids.add(target)
                    for i, elem in enumerate(node.value.elts):
                        contr, x, y, z = setup_internal_values(elem, contr, x, y, z)
                        x += 1
                        contr.add_block((x, y, z), CommandBlock(
                            'summon ArmorStand ~ ~1 ~ {{NoGravity:1b,Tags:["list_noname"]}}'
                        ))
                        x += 1
                        contr.add_block((x, y, z), CommandBlock(
                            'scoreboard players set @e[type=ArmorStand,tag=list_noname] py2cb_ids {0}'
                                .format(listids[target.id])
                        ))
                        x += 1
                        contr.add_block((x, y, z), CommandBlock(
                            'scoreboard players set @e[type=ArmorStand,tag=list_noname] py2cb_idxs {0}'.format(i)
                        ))
                        x += 1
                        contr.add_block((x, y, z), CommandBlock(
                            'scoreboard players set @e[type=ArmorStand,tag=list_noname] py2cb_var {0}'
                                .format(get_player_and_obj(elem))
                        ))
                        x += 1
                        contr.add_block((x, y, z), CommandBlock(
                            'entitydata @e[type=ArmorStand,tag=list_noname] {Tags:["list"]}'
                        ))
                
                # Not-so-simple assignment - name = op (ex: n = 2 * 3)
                # If it's an expr and it hasn't been caught yet, we assume it's a complexish expression
                elif isinstance(node.value, ast.expr):
                    contr, x, y, z = parse_node(node.value, contr, x, y, z)
                    x += 1
                    contr.add_block((x, y, z), CommandBlock(
                        'scoreboard players operation {0} py2cb_var = expr_{1} py2cb_intrnl'
                            .format(target, exprids[node.value])
                    ))
            elif isinstance(target, ast.Subscript):
                if target.value not in listids:
                    raise Exception('Cannot subscript a non-list.')
                
                if isinstance(target.slice, ast.Index):
                    contr, x, y, z = setup_internal_values(node.value, contr, x, y, z)
                    
                    if isinstance(target.slice.value, ast.Num):
                        x += 1
                        contr.add_block((x, y, z), CommandBlock(
                            'scoreboard players operation @e[type=ArmorStand,tag=list,score_py2cb_idxs={0},'
                                'score_py2cb_idxs_min={0},score_py2cb_ids={1},score_py2cb_ids_min={1}] py2cb_var = {2}'
                                .format(target.slice.value.n, listids[target.value], get_player_and_obj(node.value))
                        ))
                    else:
                        contr, x, y, z = setup_internal_values(target.slice.value, contr, x, y, z)
                        
                        x += 1
                        contr.add_block((x, y, z), CommandBlock(
                            'scoreboard players operation @e[type=ArmorStand,tag=list,score_py2cb_ids={0},'
                                'score_py2cb_ids_min={0}] py2cb_idxs -= {1}'
                                .format(listids[target.value], get_player_and_obj(target.slice.value))
                        ))
                        x += 1
                        contr.add_block((x, y, z), CommandBlock(
                            'scoreboard players operation @e[type=ArmorStand,tag=list,score_py2cb_ids={0},'
                                'score_py2cb_ids_min={0},score_py2cb_idxs=0,score_py2cb_idxs_min=0] py2cb_var = {1}'
                                .format(listids[target.value], get_player_and_obj(node.value))
                        ))
                        x += 1
                        contr.add_block((x, y, z), CommandBlock(
                            'scoreboard players operation @e[type=ArmorStand,tag=list,score_py2cb_ids={0},'
                                'score_py2cb_ids_min={0}] py2cb_idxs += {1}'
                                .format(listids[target.value], get_player_and_obj(target.slice.value))
                        ))
                else:
                    raise Exception('The only slice type supported is index (no colons allowed).')
            else:
                raise Exception('Only names and subscripts are supported as assignment targets.')
    
    # AUGMENTED ASSIGNMENTS
    elif isinstance(node, ast.AugAssign):
        if isinstance(node.target, ast.Name):
            contr, x, y, z = setup_internal_values(node.value, contr, x, y, z)
            x += 1
            contr.add_block((x, y, z), CommandBlock(
                'scoreboard players operation {0} py2cb_var {2}= {1}'
                    .format(node.target.id, get_player_and_obj(node.value), get_op_char(node.op))
            ))
        else:
            raise Exception('Only names are supported as assignment targets.')
    
    # BINOPS
    elif isinstance(node, ast.BinOp):
        for side in [node.left, node.right]:
            contr, x, y, z = setup_internal_values(side, contr, x, y, z)
        
        # <= is issubset operator on sets
        if set(map(type, [node.left, node.right])) <= {ast.Num, ast.Name}:
            x += 1
            exprids.add(node)
            contr.add_block((x, y, z), CommandBlock(
                'scoreboard players operation expr_{0} py2cb_intrnl = {1}'
                    .format(exprids[node], get_player_and_obj(node.left))
            ))
            x += 1
            contr.add_block((x, y, z), CommandBlock(
                'scoreboard players operation expr_{0} py2cb_intrnl {2}= {1}'
                    .format(exprids[node], get_player_and_obj(node.right), get_op_char(node.op))
            ))
    
    # BOOLOPS
    elif isinstance(node, ast.BoolOp):
        exprids.add(node)
        if len(node.values) > 2:
            prev = ast.BoolOp(op=node.op, values=node.values[:-1], lineno=0, col_offset=0)
            contr, x, y, z = parse_node(prev, contr, x, y, z)
            x += 1
            contr.add_block((x, y, z), CommandBlock(
                'scoreboard players operation expr_{0} py2cb_intrnl = expr_{1} py2cb_intrnl'
                    .format(exprids[node], exprids[prev])
            ))
        else:
            contr, x, y, z = setup_internal_values(node.values[0], contr, x, y, z)
            
            x += 1
            contr.add_block((x, y, z), CommandBlock(
                'scoreboard players operation expr_{0} py2cb_intrnl = {1}'
                    .format(exprids[node], get_player_and_obj(node.values[0]))
            ))
        
        contr, x, y, z = setup_internal_values(node.values[-1], contr, x, y, z)
        
        if isinstance(node.op, ast.And):
            # a and b = a * b where a and b are both 0 or 1
            x += 1
            contr.add_block((x, y, z), CommandBlock(
                'scoreboard players operation expr_{0} py2cb_intrnl *= {1}'
                    .format(exprids[node], get_player_and_obj(node.values[-1]))
            ))
        else:  # isinstance(node.op, ast.Or) - it's the only other option
            # a or b = min(a + b, 1) where a and b are both 0 or 1
            contr, x, y, z = add_const(1, contr, x, y, z)
            x += 1
            contr.add_block((x, y, z), CommandBlock(
                'scoreboard players operation expr_{0} py2cb_intrnl += {1}'
                    .format(exprids[node], get_player_and_obj(node.values[-1]))
            ))
            x += 1
            contr.add_block((x, y, z), CommandBlock(
                'scoreboard players operation expr_{0} py2cb_intrnl < const_1 py2cb_intrnl'.format(exprids[node])
            ))
    
    # UNARYOPS
    elif isinstance(node, ast.UnaryOp):
        if isinstance(node.op, ast.UAdd):
            contr, x, y, z = parse_node(node.operand, contr, x, y, z)
        else:
            contr, x, y, z = setup_internal_values(node.operand, contr, x, y, z)
            exprids.add(node)
            
            if type(node.op) in (ast.USub, ast.Invert):
                x += 1
                contr.add_block((x, y, z), CommandBlock(
                    'scoreboard players set temp py2cb_intrnl 0'
                ))
                x += 1
                contr.add_block((x, y, z), CommandBlock(
                    'scoreboard players operation temp py2cb_intrnl -= {0}'.format(get_player_and_obj(node.operand))
                ))
                x += 1
                contr.add_block((x, y, z), CommandBlock(
                    'scoreboard players operation expr_{0} = temp py2cb_intrnl'.format(exprids[node])
                ))
                
                if isinstance(node.op, ast.Invert):
                    contr, x, y, z = add_const(1, contr, x, y, z)
                    contr.add_block((x, y, z), CommandBlock(
                        'scoreboard players operation expr_{0} -= const_1 py2cb_intrnl'.format(exprids[node])
                    ))
            else:  # isinstance(node.op, ast.Not) - it's the only other option
                # Pseudocode: temp = operand; operand = 0; if temp == 0: operand = 1
                x += 1
                contr.add_block((x, y, z), CommandBlock(
                    'scoreboard players operation temp py2cb_intrnl = {0}'.format(get_player_and_obj(node.operand))
                ))
                x += 1
                contr.add_block((x, y, z), CommandBlock(
                    'scoreboard players set expr_{0} 0'.format(exprids[node])
                ))
                x += 1
                contr.add_block((x, y, z), CommandBlock('scoreboard players test temp py2cb_intrnl 0 0'))
                x += 1
                contr.add_block((x, y, z), CommandBlock(
                    'scoreboard players set expr_{0} 1'.format(exprids[node]),
                    metadata=CommandBlock.EAST | CommandBlock.CONDITIONAL
                ))
    
    # IFEXPS
    elif isinstance(node, ast.IfExp):
        # Pseudocode: res = body; if test == 0: res = orelse
        for expr in [node.body, node.orelse, node.test]:
            contr, x, y, z = setup_internal_values(expr, contr, x, y, z)
        exprids.add(node)
        
        x += 1
        contr.add_block((x, y, z), CommandBlock(
            'scoreboard players operation expr_{0} = {1}'.format(exprids[node], get_player_and_obj(node.body))
        ))
        x += 1
        contr.add_block((x, y, z), CommandBlock(
            'scoreboard players test {0} 0 0'.format(get_player_and_obj(node.orelse))
        ))
        x += 1
        contr.add_block((x, y, z), CommandBlock(
            'scoreboard players operation expr_{0} = {1}'.format(exprids[node], get_player_and_obj(node.orelse)),
            metadata=CommandBlock.EAST | CommandBlock.CONDITIONAL
        ))
    
    # COMPARES
    elif isinstance(node, ast.Compare):
        for operand in [node.left] + node.comparators:
            contr, x, y, z = setup_internal_values(operand, contr, x, y, z)
        
        left = node.left
        for op, right in zip(node.ops, node.comparators):
            current = ast.Compare(left=left, ops=[op], comparators=[right])
            exprids.add(current)
            
            if type(op) in (ast.Eq, ast.NotEq, ast.Gt, ast.GtE, ast.Lt, ast.LtE):
                x += 1
                contr.add_block((x, y, z), CommandBlock(
                    'scoreboard players operation temp py2cb_intrnl = {0}'.format(get_player_and_obj(left))
                ))
                x += 1
                contr.add_block((x, y, z), CommandBlock(
                    'scoreboard players operation temp py2cb_intrnl -= {0}'.format(get_player_and_obj(right))
                ))
                x += 1
                contr.add_block((x, y, z), CommandBlock(
                    'scoreboard players set expr_{0} py2cb_intrnl {1}'
                        .format(exprids[current], int(isinstance(op, ast.NotEq)))
                ))
                x += 1
                contr.add_block((x, y, z), CommandBlock(
                    'scoreboard players test temp py2cb_intrnl {0} {1}'.format(
                        {ast.Eq: 0, ast.NotEq: 0, ast.Gt: 1, ast.GtE: 0, ast.Lt: '*', ast.LtE: '*'}[type(op)],
                        {ast.Eq: 0, ast.NotEq: 0, ast.Gt: '*', ast.GtE: '*', ast.Lt: -1, ast.LtE: 0}[type(op)]
                    )
                ))
                x += 1
                contr.add_block((x, y, z), CommandBlock(
                    'scoreboard players set expr_{0} py2cb_intrnl {1}'
                        .format(exprids[current], int(not isinstance(op, ast.NotEq))),
                    metadata=CommandBlock.EAST | CommandBlock.CONDITIONAL
                ))
            else:
                raise Exception('Invalid comparison operation (only ==, !=, <, <=, >, and >= are allowed).')
            
            left = right
        
        exprids.add(node)
        x += 1
        # noinspection PyUnboundLocalVariable
        contr.add_block((x, y, z), CommandBlock(
            'scoreboard players operation expr_{0} py2cb_intrnl = expr_{1} py2cb_intrnl'
                .format(exprids[node], exprids[current])
        ))
    
    # SUBSCRIPTS
    elif isinstance(node, ast.Subscript):
        if isinstance(node.slice, ast.Index):
            if node.value not in listids:
                raise Exception('Cannot subscript a non-list (the reference was most likely undefined).')
            
            contr, x, y, z = setup_internal_values(node.slice.value, contr, x, y, z)
            exprids.add(node)
            
            if isinstance(node.slice.value, ast.Num):
                x += 1
                contr.add_block((x, y, z), CommandBlock(
                    'scoreboard players operation expr_{0} py2cb_intrnl = @e[type=ArmorStand,tag=list,'
                        'score_py2cb_idxs={1},score_py2cb_idxs_min={1},score_py2cb_ids={2},score_py2cb_min={2}] '
                        'py2cb_var'.format(exprids[node], listids[node.value], node.slice.value.n)
                ))
            else:
                x += 1
                contr.add_block((x, y, z), CommandBlock(
                    'scoreboard players operation @e[type=ArmorStand,tag=list,score_py2cb_ids={0},'
                        'score_py2cb_ids_min={0}] py2cb_idxs -= {1}'
                        .format(listids[node.value], get_player_and_obj(node.slice.value))
                ))
                x += 1
                contr.add_block((x, y, z), CommandBlock(
                    'scoreboard players operation expr_{0} py2cb_intrnl = @e[type=ArmorStand,tag=list,'
                        'score_py2cb_idxs=0,score_py2cb_idxs_min=0,score_py2cb_ids={1},score_py2cb_ids_min={1}] '
                        'py2cb_var'.format(exprids[node], listids[node.value])
                ))
                x += 1
                contr.add_block((x, y, z), CommandBlock(
                    'scoreboard players operation @e[type=ArmorStand,tag=list,score_py2cb_idxs=0,'
                        'score_py2cb_idxs_min=0,score_py2cb_ids={1},score_py2cb_ids_min={1}] py2cb_idxs += {0}'
                        .format(get_player_and_obj(node.slice.value), listids[node.value])
                ))
        else:
            raise Exception('The only slice type supported is index (no colons allowed).')
    
    # IF STATEMENTS
    elif isinstance(node, ast.If):
        contr, x, y, z = setup_internal_values(node.test, contr, x, y, z)
        x += 1
        contr.add_block((x, y, z), CommandBlock(
            'scoreboard players test {0} 0 0'.format(get_player_and_obj(node.test))
        ))
        num_branches += 1
        contr, x, y, z = add_pulsegiver_block(contr, x, y, z)
        x += 1
        contr.add_block((x, y, z), CommandBlock(
            'scoreboard players test {0} * -1'.format(get_player_and_obj(node.test))
        ))
        num_branches += 1
        contr, x, y, z = add_pulsegiver_block(contr, x, y, z)
        x += 1
        contr.add_block((x, y, z), CommandBlock(
            'scoreboard players test {0} 1 *'.format(get_player_and_obj(node.test))
        ))
        contr, x, y, z = add_pulsegiver_block(contr, x, y, z)
        x += 1
        
        # if body block
        xyz = x, y, z
        x = 0
        z = num_branches - 1
        for stmt in node.body:
            contr, x, y, z = parse_node(stmt, contr, x, y, z)
        contr, x, y, z = add_pulsegiver_block(contr, x, y, z, *xyz)
        
        # else body block
        x = 0
        z = num_branches - 2
        for stmt in node.orelse:
            contr, x, y, z = parse_node(stmt, contr, x, y, z)
        contr, x, y, z = add_pulsegiver_block(contr, x, y, z, *xyz)
        
        x, y, z = xyz
    
    # WHILE LOOPS
    elif isinstance(node, ast.While):
        # There's an if statement to go to the while loop, and one at the end of the while loop
        # 'else' on while loops/break/continue unsupported
        if node.orelse:
            raise Exception('else statement on while loop is not supported')
        
        contr, x, y, z = setup_internal_values(node.test, contr, x, y, z)
        
        x += 1
        num_branches += 1
        contr.add_block((x, y, z), CommandBlock(
            'scoreboard players test {0} * -1'.format(get_player_and_obj(node.test))
        ))
        contr, x, y, z = add_pulsegiver_block(contr, x, y, z)
        x += 1
        contr.add_block((x, y, z), CommandBlock(
            'scoreboard players test {0} 1 *'.format(get_player_and_obj(node.test))
        ))
        contr, x, y, z = add_pulsegiver_block(contr, x, y, z)
        x += 1
        
        # while body branch
        xyz = x, y, z
        x = 0
        z = num_branches - 1
        for stmt in node.body:
            contr, x, y, z = parse_node(stmt, contr, x, y, z)
        contr, x, y, z = setup_internal_values(node.test, contr, x, y, z, redef=True)
        x += 1
        contr.add_block((x, y, z), CommandBlock(
            'scoreboard players test {0} 0 0'.format(get_player_and_obj(node.test))
        ))
        contr, x, y, z = add_pulsegiver_block(contr, x, y, z, *xyz)  # gives control back to while caller
        x += 1
        contr.add_block((x, y, z), CommandBlock(
            'scoreboard players test {0} * -1'.format(get_player_and_obj(node.test))
        ))
        contr, x, y, z = add_pulsegiver_block(contr, x, y, z)  # gives pulse to own branch
        x += 1
        contr.add_block((x, y, z), CommandBlock(
            'scoreboard players test {0} 1 *'.format(get_player_and_obj(node.test))
        ))
        contr, x, y, z = add_pulsegiver_block(contr, x, y, z)
        
        x, y, z = xyz
    
    # BREAK/CONTINUE - not supported
    elif type(node) in (ast.Break, ast.Continue):
        raise Exception('break/continue are not supported.')
    
    # BARE EXPRs
    elif isinstance(node, ast.Expr):
        contr, x, y, z = parse_node(node.value, contr, x, y, z)
    
    return contr, x, y, z


def parse(ast_root: ast.Module) -> Contraption:
    res = Contraption()
    x = y = z = 0
    res.add_block((x, y, z), CommandBlock('scoreboard objectives add py2cb_intrnl dummy Py2CB Internal Variables',
                                          type_=CommandBlock.IMPULSE, auto=False))
    x += 1
    res.add_block((x, y, z), CommandBlock('scoreboard objectives add py2cb_var dummy Py2CB Application Variables'))
    x += 1
    res.add_block((x, y, z), CommandBlock('scoreboard objectives add py2cb_ids dummy Py2CB IDs'))
    x += 1
    res.add_block((x, y, z), CommandBlock('scoreboard objectives add py2cb_idxs dummy Py2CB Indexes'))
    
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
