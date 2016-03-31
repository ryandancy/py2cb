#!usr/bin/env python
# -*- coding: utf-8 -*-

import ast
import unicodedata
import py2cb.script as script

from pynbt import NBTFile, TAG_Byte, TAG_Byte_Array, TAG_Compound, TAG_Int, TAG_List, TAG_Short, TAG_String
from typing import Tuple, List, Dict, Any, Optional, Sequence, Union, Callable

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
    
    def __init__(self, command: str, type_: int = CHAIN, direction: int = EAST, conditional: bool = False,
                 auto: bool = True) -> None:
        self.command = command
        self.type_ = type_
        self.metadata = direction | (CommandBlock.CONDITIONAL * conditional)
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
    
    def get_gen_command(self, offx: int, offz: int) -> str:
        return 'setblock ~{0} ~ ~{1} minecraft:{2} {3} replace {{Command:"{4}",auto:{5}b}}'.format(
            offx, offz, self.get_command_block_name(), self.metadata, self.command, int(self.auto)
        )
    
    def get_dump(self) -> List[str]:
        """Generates a list of Type, Metadata, Auto, Commmand. Mostly exists for Contraption.get_dump."""
        return [self.get_command_block_name(), self.metadata, self.auto, self.command]


class Contraption:
    
    def __init__(self) -> None:
        self.cblocks = []  # type: List[Tuple[Tuple[int, int], CommandBlock]]
    
    def add_block(self, xz: Tuple[int, int], block: CommandBlock) -> None:
        self.cblocks.append((xz, block))
    
    def get_dump(self) -> List[List[str]]:
        """Generates a table of command block x, z, Type, Metadata, Auto, Command."""
        header = ['X', 'Z', 'Type', 'Metadata', 'Auto', 'Command']
        res = [header]
        for xz, cblock in self.cblocks:
            res.append(list(xz) + cblock.get_dump())
        return res
    
    def get_schematic(self) -> NBTFile:
        """
        Exports this contraption to a schematic NBT file.
        Remember to pass compression=NBTFile.Compression.GZIP when calling save!
        """
        # Uses unofficial .schematic format found at the Minecraft Wiki (minecraft.gamepedia.com/Schematic_file_format)
        # (ish, because MCEdit uses a 'Biomes' TAG_Byte_Array not documented on the wiki)
        nbt = NBTFile(name='Schematic')
        
        width = max(x for (x, z), cblock in self.cblocks) + 1
        height = 1
        length = max(z for (x, z), cblock in self.cblocks) + 1
        
        # blocks and data are sorted by height/, then length/z, then width/x (YZX)
        # therefore the index of x, z in blocks/data is (y * length + z) * width + x
        blocks = [0] * (width * length)  # 0 is air
        data = [0] * (width * length)
        tiles = []
        for (x, z), cblock in self.cblocks:
            index = z * width + x  # y is always 0, so (y * length + z) * width + x reduces to this
            blocks[index] = cblock.get_block_id()
            data[index] = cblock.metadata
            
            tiles.append(TAG_Compound(value={
                'x': TAG_Int(x),
                'z': TAG_Int(z),
                'Command': TAG_String(cblock.command),
                'auto': TAG_Byte(int(cblock.auto)),
                # everything below here is the same for every cblock
                'y': TAG_Int(0),
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
    
    def __init__(self, has_limit=False) -> None:
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
    
    def add(self, var: Any, id_: Optional[int] = None) -> None:
        # If var is added to multiple times it is silently overriden
        if id_ is None:
            id_ = self._next_id()
        self._vars_to_ids[var] = id_
    
    def __getitem__(self, var: Any) -> int:
        return self._vars_to_ids[var]
    
    def __contains__(self, var: Any) -> bool:
        return var in self._vars_to_ids


stringids = IDContainer(has_limit=True)
exprids = IDContainer()
listids = IDContainer(has_limit=True)
scopeids = IDContainer()  # limit is implemented in Scope.__init__


class Scope:
    
    names_to_scopes = {}  # type: Dict[str, Scope]
    _used_ids = []
    
    def __init__(self, name: str, node: Optional[ast.FunctionDef], parent: Optional['Scope']) -> None:
        self.children = []
        
        # None parent means that this scope is the global scope
        if parent is not None:
            self.parent = parent
            self.parent.children.append(self)
        
        scopeids.add(self)
        self.id = scopeids[self]
        
        i = self.id
        while chr(i) == '"' or i in Scope._used_ids or unicodedata.category(chr(i))[0] in 'CZ':
            # We don't want control characters/format/not assigned/surrogates/private use/whitespace in suffixes
            # '"' is the quote char in MC JSON
            i += 1
            
            if i > 0xFFFF:
                raise Exception('Too many scopes!')
        
        Scope._used_ids.append(i)
        self.char = chr(i)
        
        self.node = node
        self.name = name
        Scope.names_to_scopes[self.name] = self
    
    def transform(self, name: str) -> str:
        return name.ljust(39, '-') + self.char if len(name) != 40 else name

Scope.GLOBAL = Scope('<global>', None, None)


consts = []
num_branches = 1


def add_const(const: int, contr: Contraption, x: int, z: int) -> Tuple[Contraption, int, int]:
    if const not in consts:
        x += 1
        contr.add_block((x, z), CommandBlock('scoreboard players set const_{0} py2cb_intrnl {0}'.format(const)))
        consts.append(const)
    return contr, x, z


def get_player_and_obj(node: ast.AST, scope: Scope) -> str:
    """Assumes that all relavant things are in place (const_n, etc)"""
    if isinstance(node, ast.Num):
        return 'const_{0} py2cb_intrnl'.format(node.n)
    elif isinstance(node, ast.Name):
        node.id = scope.transform(node.id)
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


def setup_internal_values(node: ast.AST, scope: Scope, contr: Contraption, x: int, z: int, redef: bool = False) \
        -> Tuple[ast.AST, Contraption, int, int]:
    if isinstance(node, ast.Num):
        contr, x, z = add_const(node.n, contr, x, z)
    elif isinstance(node, ast.NameConstant):
        contr, x, z = add_const(nameconstant_to_int(node), contr, x, z)
    elif isinstance(node, ast.Name):
        node.id = scope.transform(node.id)
    elif redef or node not in exprids:
        contr, x, z = parse_node(node, scope, contr, x, z)
        if node not in exprids:
            exprids.add(node)
    return node, contr, x, z


def add_pulsegiver_block(contr: Contraption, x: int, z: int,
                         wx: Optional[int] = None, wz: Optional[int] = None, conditional: bool = True) \
        -> Tuple[Contraption, int, int]:
    """(wx, wz) defaults to (-x - 1, num_branches - z - 1)"""
    if wx is None:
        offx = -x - 1
    else:
        offx = wx - x - 1
    if wz is None:
        offz = num_branches - z - 1
    else:
        offz = wz - z
    
    x += 1
    contr.add_block((x, z), CommandBlock(
        CommandBlock(
            'setblock ~ ~ ~ minecraft:air', type_=CommandBlock.IMPULSE, auto=True
        ).get_gen_command(offx, offz),
        conditional=conditional
    ))
    
    return contr, x, z


def get_json(node: ast.AST, scope: Scope, style: Optional[int] = None) -> str:
    """Assumes setup_internal_values has been called"""
    if isinstance(node, ast.Num):
        json = '"score":{{"name":"const_{0}","objective":"py2cb_intrnl"}}'.format(node.n)
    elif isinstance(node, ast.Str):
        json = '"text":"{0}"'.format(node.s)
    elif isinstance(node, ast.NameConstant):
        json = '"score":{{"name":"const_{0}","objective":"py2cb_intrnl"}}'.format(nameconstant_to_int(node))
    elif isinstance(node, ast.Name):
        node.id = scope.transform(node.id)
        if node.id in stringids:
            json = '"selector":"@e[type=ArmorStand,tag=string,score_py2cb_var={0},score_py2cb_var_min={0}]"' \
                   .format(stringids[node.id])
        else:
            json = '"score":{{"name":"{0}","objective":"py2cb_var"}}'.format(node.id)
    elif node in exprids:
        json = '"score":{{"name":"expr_{0}","objective":"py2cb_intrnl"}}'.format(exprids[node])
    else:
        raise Exception('Only nums/strs/true/false/names/expressions are allowed in tellraw/other JSON.')
    
    if style:
        json += ',' + ','.join(script.parse_style_bitmap(
            style, lambda name, value: '"{0}":"{1}"'.format(name,
                                                            script.colours_to_mc[value] if name == 'color' else 'true')
        ))
    
    return '{' + json + '}'


def is_bitmap_safe(bitmap: ast.AST) -> bool:
    if isinstance(bitmap, ast.Num) or isinstance(bitmap, ast.Name):
        return True
    
    if isinstance(bitmap, ast.BinOp):
        if not isinstance(bitmap.op, ast.BitOr):
            return False
        return is_bitmap_safe(bitmap.left) and is_bitmap_safe(bitmap.right)
    
    return False


def parse_kwargs(keywords: Sequence[ast.keyword], allowed: Sequence[Tuple[str, Any, Sequence[type]]], method_name: str)\
        -> Dict[str, Any]:
    """
    allowed is Sequence[Tuple[name, default: allowable_type, Sequence[ast_subtype_literal]]]
    return type is Dict[str, allowable_type]
    """
    allowed_names, defaults, allowed_types = zip(*allowed)
    
    if len(keywords) > len(allowed):
        raise Exception('Only {0} keyword argument{1} ({2}) {3} allowed on {4}().'.format(
            len(allowed), '' if len(allowed) == 1 else 's', ', '.join(allowed_names),
            'is' if len(allowed) == 1 else 'are', method_name
        ))
    
    elif keywords:
        res = {}
        
        for keyword in keywords:
            if keyword.arg not in allowed_names:
                raise Exception('Illegal keyword argument on {0}(): {1}.'.format(method_name, keyword.arg))
            
            index = allowed_names.index(keyword.arg)
            
            if type(keyword.value) not in allowed_types[index]:
                raise Exception('The value of {0} in {1}() must be one of {2}.'.format(
                    keyword.arg, method_name, ','.join(atype.__name__ for atype in allowed_types[index])
                ))
            
            res[keyword.arg] = keyword.value
    
    else:
        res = {}
    
    for unspecified_kwarg_name in set(allowed_names) - set(res.keys()):
        res[unspecified_kwarg_name] = defaults[allowed_names.index(unspecified_kwarg_name)]
    
    return res


def get_func_name(node: ast.Call) -> str:
    # spam()
    if isinstance(node.func, ast.Name):
        return node.func.id
    
    # spam.ham()
    elif isinstance(node.func, ast.Attribute):
        return node.func.attr
    
    # Something illegal like spam()()
    else:
        raise Exception('Illegal function call.')


def func_call_matches(call: ast.Call, func: ast.FunctionDef) -> bool:
    # We assume that the name has already been checked and the function sig is valid, and perform no type-hint checking
    # *args/**kwargs are not supported, self/cls are not accounted for
    # Make sure that all missing args are not required args
    call_kwarg_names = [kwarg.arg for kwarg in call.keywords]
    num_pos_args = len(call.args)
    for arg in func.args.args:
        if arg.arg in call_kwarg_names:
            del call_kwarg_names[call_kwarg_names.index(arg.arg)]
        elif num_pos_args > 0:
            num_pos_args -= 1
        elif arg not in func.args.args[-len(func.args.defaults):]:
            return False
    return True


# Merge with func_call_matches???
def get_arg_dict(call: ast.Call, func: ast.FunctionDef) -> Dict[str, ast.AST]:
    """assumes that func_call_matches(call, func)"""
    res = {}
    
    kwarg_dict = {kwarg.arg: kwarg.value for kwarg in call.keywords}
    arg_names = [arg.arg for arg in func.args.args if arg.arg not in kwarg_dict and
                 (len(kwarg_dict) == 0 or len(func.args.defaults) == 0 or
                  arg not in func.args.args[-len(func.args.defaults):])]
    call_args = call.args
    for arg in func.args.args:
        if arg.arg in kwarg_dict:
            res[arg.arg] = kwarg_dict[arg.arg]
        elif arg.arg in arg_names and call_args:
            # arg.arg should be arg_names[0]...
            res[arg.arg] = call_args[0]
            del call_args[0]
            del arg_names[arg_names.index(arg.arg)]
        elif arg in func.args.args[-len(func.args.defaults):]:
            res[arg.arg] = func.args.defaults[len(func.args.args) - func.args.args.index(arg) - 1]
    
    return res


Parser = Callable[[ast.AST, Scope, Contraption, int, int], Tuple[Contraption, int, int]]
ast_to_parsers = {}  # type: Dict[type, Parser]


def parser(*types: Sequence[type]):
    def inner(func: Parser):
        for type_ in types:
            ast_to_parsers[type_] = func
        return func
    return inner


def parse_node(node: ast.AST, scope: Scope, contr: Contraption, x: int, z: int) -> Tuple[Contraption, int, int]:
    try:
        node_parser = ast_to_parsers[type(node)]
    except KeyError:
        # Silently ignore things like pass, docstrings, etc
        return contr, x, z
    
    return node_parser(node, scope, contr, x, z)


@parser(ast.Assign)
def parse_assignment(node: ast.Assign, scope: Scope, contr: Contraption, x: int, z: int) \
        -> Tuple[Contraption, int, int]:
    for target in node.targets:
        # Assignment with names (n = _)
        if isinstance(target, ast.Name):
            # Minecraft prevents player names >40 characters long, we reserve the last char
            if len(target.id) > 39:
                raise Exception('Max name length is 39 chars (Minecraft caps at 40, Py2CB reserves last).')
            
            target.id = scope.transform(target.id)
            
            # Simple assignment - name = num (ex: n = 4)
            if isinstance(node.value, ast.Num):
                x += 1
                contr.add_block((x, z), CommandBlock(
                    'scoreboard players set {0} py2cb_var {1}'.format(target.id, node.value.n)
                ))
            
            # Simple assignment - name = str (ex: n = 'foo')
            elif isinstance(node.value, ast.Str):
                # A Minecraft limitation (bugfix for MC-78862) prevents entities being named ""
                # Auto-set to placeholder string???
                if node.value.s == '':
                    raise Exception('Due to a Minecraft limitation, the empty string cannot be used in entity '
                                    'names, which is how string variables are stored. Please don\'t assign your '
                                    'string variable to "". Sorry about that.')
                
                # Strings are represented by armor stands with custom names
                if target.id not in stringids:
                    stringids.add(target.id)
                
                x += 1
                contr.add_block((x, z), CommandBlock(
                    'kill @e[type=ArmorStand,tag=string,score_py2cb_var={0},score_py2cb_var_min={0}]'
                        .format(stringids[target.id])
                ))
                x += 1
                contr.add_block((x, z), CommandBlock(
                    'summon ArmorStand ~ ~1 ~ {{NoGravity:1b,CustomName:"{0}",Tags:["string_noname","py2cb"]}}'
                        .format(node.value.s)
                ))
                
                x += 1
                contr.add_block((x, z), CommandBlock(
                    'scoreboard players set @e[type=ArmorStand,tag=string_noname] py2cb_var {0}'
                        .format(stringids[target.id])
                ))
                
                x += 1
                contr.add_block((x, z), CommandBlock(
                    'entitydata @e[type=ArmorStand,tag=string_noname] {Tags:["string","py2cb"]}'
                ))
            
            # Simple assignment - name = True/False/None (ex: n = True)
            elif isinstance(node.value, ast.NameConstant):
                x += 1
                contr.add_block((x, z), CommandBlock(
                    'scoreboard players set {0} py2cb_var {1}'
                        .format(target.id, nameconstant_to_int(node.value))
                ))
            
            # Simple assignment - name = list (ex: n = [5, 6, 7])
            # TODO Strings in lists
            elif isinstance(node.value, ast.List):
                if target.id not in listids:
                    listids.add(target.id)
                
                for i, elem in enumerate(node.value.elts):
                    elem, contr, x, z = setup_internal_values(elem, scope, contr, x, z)
                    x += 1
                    contr.add_block((x, z), CommandBlock(
                        'kill @e[type=ArmorStand,tag=list,score_py2cb_ids={0},score_py2cb_ids_min={0},'
                        'score_py2cb_idxs={1},score_py2cb_idxs_min={1}]'.format(listids[target.id], i)
                    ))
                    x += 1
                    contr.add_block((x, z), CommandBlock(
                        'summon ArmorStand ~ ~1 ~ {NoGravity:1b,Tags:["list_noname","py2cb"]}'
                    ))
                    x += 1
                    contr.add_block((x, z), CommandBlock(
                        'scoreboard players set @e[type=ArmorStand,tag=list_noname] py2cb_ids {0}'
                            .format(listids[target.id])
                    ))
                    x += 1
                    contr.add_block((x, z), CommandBlock(
                        'scoreboard players set @e[type=ArmorStand,tag=list_noname] py2cb_idxs {0}'.format(i)
                    ))
                    x += 1
                    contr.add_block((x, z), CommandBlock(
                        'scoreboard players operation @e[type=ArmorStand,tag=list_noname] py2cb_var = {0}'
                            .format(get_player_and_obj(elem, scope))
                    ))
                    x += 1
                    contr.add_block((x, z), CommandBlock(
                        'entitydata @e[type=ArmorStand,tag=list_noname] {Tags:["list","py2cb"]}'
                    ))
            
            # Not-so-simple assignment - name = expr-or-something (ex: n = 2 * 3)
            else:
                node.value, contr, x, z = setup_internal_values(node.value, scope, contr, x, z)
                x += 1
                contr.add_block((x, z), CommandBlock(
                    'scoreboard players operation {0} py2cb_var = {1}'
                        .format(target.id, get_player_and_obj(node.value, scope))
                ))
        elif isinstance(target, ast.Subscript):
            if not isinstance(target.value, ast.Name):
                raise Exception('Only names can be subscripted at this time.')
            
            target.value.id = scope.transform(target.value.id)
            
            if target.value.id not in listids:
                raise Exception('Cannot subscript a non-list.')
            
            if isinstance(target.slice, ast.Index):
                node.value, contr, x, z = setup_internal_values(node.value, scope, contr, x, z)
                
                if isinstance(target.slice.value, ast.Num):
                    x += 1
                    contr.add_block((x, z), CommandBlock(
                        'scoreboard players operation @e[type=ArmorStand,tag=list,score_py2cb_idxs={0},'
                        'score_py2cb_idxs_min={0},score_py2cb_ids={1},score_py2cb_ids_min={1}] py2cb_var = {2}'
                            .format(target.slice.value.n, listids[target.value.id],
                                    get_player_and_obj(node.value, scope))
                    ))
                else:
                    target.slice.value, contr, x, z = setup_internal_values(target.slice.value, scope, contr, x, z)
                    
                    x += 1
                    contr.add_block((x, z), CommandBlock(
                        'scoreboard players operation @e[type=ArmorStand,tag=list,score_py2cb_ids={0},'
                        'score_py2cb_ids_min={0}] py2cb_idxs -= {1}'
                            .format(listids[target.value.id], get_player_and_obj(target.slice.value, scope))
                    ))
                    x += 1
                    contr.add_block((x, z), CommandBlock(
                        'scoreboard players operation @e[type=ArmorStand,tag=list,score_py2cb_ids={0},'
                        'score_py2cb_ids_min={0},score_py2cb_idxs=0,score_py2cb_idxs_min=0] py2cb_var = {1}'
                            .format(listids[target.value.id], get_player_and_obj(node.value, scope))
                    ))
                    x += 1
                    contr.add_block((x, z), CommandBlock(
                        'scoreboard players operation @e[type=ArmorStand,tag=list,score_py2cb_ids={0},'
                        'score_py2cb_ids_min={0}] py2cb_idxs += {1}'
                            .format(listids[target.value.id], get_player_and_obj(target.slice.value, scope))
                    ))
            else:
                raise Exception('The only slice type supported is index (no colons allowed).')
        else:
            raise Exception('Only names and subscripts are supported as assignment targets.')
    
    return contr, x, z


@parser(ast.AugAssign)
def parse_aug_assignment(node: ast.AugAssign, scope: Scope, contr: Contraption, x: int, z: int) \
        -> Tuple[Contraption, int, int]:
    if isinstance(node.target, ast.Name):
        node.target.id = scope.transform(node.target.id)
        
        node.value, contr, x, z = setup_internal_values(node.value, scope, contr, x, z)
        x += 1
        contr.add_block((x, z), CommandBlock(
            'scoreboard players operation {0} py2cb_var {2}= {1}'
                .format(node.target.id, get_player_and_obj(node.value, scope), get_op_char(node.op))
        ))
    else:
        raise Exception('Only names are supported as assignment targets.')
    
    return contr, x, z


@parser(ast.BinOp)
def parse_binop(node: ast.BinOp, scope: Scope, contr: Contraption, x: int, z: int) -> Tuple[Contraption, int, int]:
    for side in [node.left, node.right]:
        side, contr, x, z = setup_internal_values(side, scope, contr, x, z)
    
    # <= is issubset operator on sets
    if set(map(type, [node.left, node.right])) <= {ast.Num, ast.Name}:
        x += 1
        exprids.add(node)
        contr.add_block((x, z), CommandBlock(
            'scoreboard players operation expr_{0} py2cb_intrnl = {1}'
                .format(exprids[node], get_player_and_obj(node.left, scope))
        ))
        x += 1
        contr.add_block((x, z), CommandBlock(
            'scoreboard players operation expr_{0} py2cb_intrnl {2}= {1}'
                .format(exprids[node], get_player_and_obj(node.right, scope), get_op_char(node.op))
        ))
    
    return contr, x, z


@parser(ast.BoolOp)
def parse_boolop(node: ast.BoolOp, scope: Scope, contr: Contraption, x: int, z: int) -> Tuple[Contraption, int, int]:
    exprids.add(node)
    
    if len(node.values) > 2:
        prev = ast.BoolOp(op=node.op, values=node.values[:-1], lineno=0, col_offset=0)
        contr, x, z = parse_node(prev, scope, contr, x, z)
        x += 1
        contr.add_block((x, z), CommandBlock(
            'scoreboard players operation expr_{0} py2cb_intrnl = expr_{1} py2cb_intrnl'
                .format(exprids[node], exprids[prev])
        ))
    else:
        node.values[0], contr, x, z = setup_internal_values(node.values[0], scope, contr, x, z)
        
        x += 1
        contr.add_block((x, z), CommandBlock(
            'scoreboard players operation expr_{0} py2cb_intrnl = {1}'
                .format(exprids[node], get_player_and_obj(node.values[0], scope))
        ))
    
    node.values[-1], contr, x, z = setup_internal_values(node.values[-1], scope, contr, x, z)
    
    if isinstance(node.op, ast.And):
        # a and b = a * b where a and b are both 0 or 1
        x += 1
        contr.add_block((x, z), CommandBlock(
            'scoreboard players operation expr_{0} py2cb_intrnl *= {1}'
                .format(exprids[node], get_player_and_obj(node.values[-1], scope))
        ))
    else:  # isinstance(node.op, ast.Or) - it's the only other option
        # a or b = min(a + b, 1) where a and b are both 0 or 1
        contr, x, z = add_const(1, contr, x, z)
        x += 1
        contr.add_block((x, z), CommandBlock(
            'scoreboard players operation expr_{0} py2cb_intrnl += {1}'
                .format(exprids[node], get_player_and_obj(node.values[-1], scope))
        ))
        x += 1
        contr.add_block((x, z), CommandBlock(
            'scoreboard players operation expr_{0} py2cb_intrnl < const_1 py2cb_intrnl'.format(exprids[node])
        ))
    
    return contr, x, z


@parser(ast.UnaryOp)
def parse_unaryop(node: ast.UnaryOp, scope: Scope, contr: Contraption, x: int, z: int) -> Tuple[Contraption, int, int]:
    if isinstance(node.op, ast.UAdd):
        contr, x, z = parse_node(node.operand, scope, contr, x, z)
    else:
        node.operand, contr, x, z = setup_internal_values(node.operand, scope, contr, x, z)
        exprids.add(node)
        
        if type(node.op) in (ast.USub, ast.Invert):
            x += 1
            contr.add_block((x, z), CommandBlock(
                'scoreboard players set temp py2cb_intrnl 0'
            ))
            x += 1
            contr.add_block((x, z), CommandBlock(
                'scoreboard players operation temp py2cb_intrnl -= {0}'.format(get_player_and_obj(node.operand, scope))
            ))
            x += 1
            contr.add_block((x, z), CommandBlock(
                'scoreboard players operation expr_{0} = temp py2cb_intrnl'.format(exprids[node])
            ))
            
            if isinstance(node.op, ast.Invert):
                contr, x, z = add_const(1, contr, x, z)
                contr.add_block((x, z), CommandBlock(
                    'scoreboard players operation expr_{0} -= const_1 py2cb_intrnl'.format(exprids[node])
                ))
        else:  # isinstance(node.op, ast.Not) - it's the only other option
            # Pseudocode: temp = operand; operand = 0; if temp == 0: operand = 1
            x += 1
            contr.add_block((x, z), CommandBlock(
                'scoreboard players operation temp py2cb_intrnl = {0}'.format(get_player_and_obj(node.operand, scope))
            ))
            x += 1
            contr.add_block((x, z), CommandBlock(
                'scoreboard players set expr_{0} 0'.format(exprids[node])
            ))
            x += 1
            contr.add_block((x, z), CommandBlock('scoreboard players test temp py2cb_intrnl 0 0'))
            x += 1
            contr.add_block((x, z), CommandBlock(
                'scoreboard players set expr_{0} 1'.format(exprids[node]),
                conditional=True
            ))
    
    return contr, x, z


@parser(ast.IfExp)
def parse_ifexpr(node: ast.IfExp, scope: Scope, contr: Contraption, x: int, z: int) -> Tuple[Contraption, int, int]:
    # Pseudocode: res = body; if test == 0: res = orelse
    for expr in [node.body, node.orelse, node.test]:
        expr, contr, x, z = setup_internal_values(expr, scope, contr, x, z)
    
    exprids.add(node)
    x += 1
    contr.add_block((x, z), CommandBlock(
        'scoreboard players operation expr_{0} = {1}'.format(exprids[node], get_player_and_obj(node.body, scope))
    ))
    x += 1
    contr.add_block((x, z), CommandBlock(
        'scoreboard players test {0} 0 0'.format(get_player_and_obj(node.orelse, scope))
    ))
    x += 1
    contr.add_block((x, z), CommandBlock(
        'scoreboard players operation expr_{0} = {1}'.format(exprids[node], get_player_and_obj(node.orelse, scope)),
        conditional=True
    ))
    
    return contr, x, z


@parser(ast.Compare)
def parse_compare(node: ast.Compare, scope: Scope, contr: Contraption, x: int, z: int) -> Tuple[Contraption, int, int]:
    for operand in [node.left] + node.comparators:
        operand, contr, x, z = setup_internal_values(operand, scope, contr, x, z)
    
    compare_exprids = []
    
    left = node.left
    for op, right in zip(node.ops, node.comparators):
        current = (left, right)
        exprids.add(current)
        compare_exprids.append(exprids[current])
        
        if type(op) in (ast.Eq, ast.NotEq, ast.Gt, ast.GtE, ast.Lt, ast.LtE):
            x += 1
            contr.add_block((x, z), CommandBlock(
                'scoreboard players operation temp py2cb_intrnl = {0}'.format(get_player_and_obj(left, scope))
            ))
            x += 1
            contr.add_block((x, z), CommandBlock(
                'scoreboard players operation temp py2cb_intrnl -= {0}'.format(get_player_and_obj(right, scope))
            ))
            x += 1
            contr.add_block((x, z), CommandBlock(
                'scoreboard players set expr_{0} py2cb_intrnl {1}'
                    .format(exprids[current], int(isinstance(op, ast.NotEq)))
            ))
            x += 1
            contr.add_block((x, z), CommandBlock(
                'scoreboard players test temp py2cb_intrnl {0} {1}'.format(
                    {ast.Eq: 0, ast.NotEq: 0, ast.Gt: 1, ast.GtE: 0, ast.Lt: '*', ast.LtE: '*'}[type(op)],
                    {ast.Eq: 0, ast.NotEq: 0, ast.Gt: '*', ast.GtE: '*', ast.Lt: -1, ast.LtE: 0}[type(op)]
                )
            ))
            x += 1
            contr.add_block((x, z), CommandBlock(
                'scoreboard players set expr_{0} py2cb_intrnl {1}'
                    .format(exprids[current], int(not isinstance(op, ast.NotEq))),
                conditional=True
            ))
        else:
            raise Exception('Invalid comparison operation (only ==, !=, <, <=, >, and >= are allowed).')
        
        left = right
    
    if len(compare_exprids) > 1:
        for leftid, rightid in zip(compare_exprids, compare_exprids[1:]):
            x += 1
            contr.add_block((x, z), CommandBlock(
                'scoreboard players operation expr_{0} py2cb_intrnl *= expr_{1} py2cb_intrnl'.format(rightid, leftid)
            ))
    
    # noinspection PyUnboundLocalVariable
    exprids.add(node, exprids[current])
    
    return contr, x, z


@parser(ast.Subscript)
def parse_subscript(node: ast.Subscript, scope: Scope, contr: Contraption, x: int, z: int) \
        -> Tuple[Contraption, int, int]:
    if isinstance(node.slice, ast.Index):
        if not isinstance(node.value, ast.Name):
            raise Exception('Only names can be subscripted at this time, sorry.')
        
        node.value.id = scope.transform(node.value.id)
        
        if node.value.id not in listids:
            raise Exception('Cannot subscript a non-list (the reference was most likely undefined).')
        
        node.slice.value, contr, x, z = setup_internal_values(node.slice.value, scope, contr, x, z)
        exprids.add(node)
        
        if isinstance(node.slice.value, ast.Num):
            x += 1
            contr.add_block((x, z), CommandBlock(
                'scoreboard players operation expr_{0} py2cb_intrnl = @e[type=ArmorStand,tag=list,'
                'score_py2cb_idxs={1},score_py2cb_idxs_min={1},score_py2cb_ids={2},score_py2cb_min={2}] '
                'py2cb_var'.format(exprids[node], listids[node.value.id], node.slice.value.n)
            ))
        else:
            x += 1
            contr.add_block((x, z), CommandBlock(
                'scoreboard players operation @e[type=ArmorStand,tag=list,score_py2cb_ids={0},'
                'score_py2cb_ids_min={0}] py2cb_idxs -= {1}'
                    .format(listids[node.value.id], get_player_and_obj(node.slice.value, scope))
            ))
            x += 1
            contr.add_block((x, z), CommandBlock(
                'scoreboard players operation expr_{0} py2cb_intrnl = @e[type=ArmorStand,tag=list,'
                'score_py2cb_idxs=0,score_py2cb_idxs_min=0,score_py2cb_ids={1},score_py2cb_ids_min={1}] '
                'py2cb_var'.format(exprids[node], listids[node.value.id])
            ))
            x += 1
            contr.add_block((x, z), CommandBlock(
                'scoreboard players operation @e[type=ArmorStand,tag=list,score_py2cb_ids={1},score_py2cb_ids_min={1}] '
                'py2cb_idxs += {0}'.format(get_player_and_obj(node.slice.value, scope), listids[node.value.id])
            ))
    else:
        raise Exception('The only slice type supported is index (no colons allowed).')
    
    return contr, x, z


@parser(ast.If)
def parse_if_statement(node: ast.If, scope: Scope, contr: Contraption, x: int, z: int) -> Tuple[Contraption, int, int]:
    global num_branches
    
    node.test, contr, x, z = setup_internal_values(node.test, scope, contr, x, z)
    x += 1
    contr.add_block((x, z), CommandBlock(
        'scoreboard players test {0} 0 0'.format(get_player_and_obj(node.test, scope))
    ))
    num_branches += 1
    contr, x, z = add_pulsegiver_block(contr, x, z)
    x += 1
    contr.add_block((x, z), CommandBlock(
        'scoreboard players test {0} * -1'.format(get_player_and_obj(node.test, scope))
    ))
    num_branches += 1
    contr, x, z = add_pulsegiver_block(contr, x, z)
    x += 1
    contr.add_block((x, z), CommandBlock(
        'scoreboard players test {0} 1 *'.format(get_player_and_obj(node.test, scope))
    ))
    contr, x, z = add_pulsegiver_block(contr, x, z)
    x += 1
    
    # if body block
    xz = x, z
    x = 0
    z = num_branches - 1
    old_z = z
    
    for stmt in node.body:
        contr, x, z = parse_node(stmt, scope, contr, x, z)
    
    contr, x, z = add_pulsegiver_block(contr, x, z, *xz, conditional=False)
    
    # else body block
    x = 0
    z = old_z - 1
    
    for stmt in node.orelse:
        contr, x, z = parse_node(stmt, scope, contr, x, z)
    
    contr, x, z = add_pulsegiver_block(contr, x, z, *xz, conditional=False)
    
    x, z = xz
    
    return contr, x, z


@parser(ast.While)
def parse_while_loop(node: ast.While, scope: Scope, contr: Contraption, x: int, z: int) -> Tuple[Contraption, int, int]:
    global num_branches
    
    # There's an if statement to go to the while loop, and one at the end of the while loop
    # 'else' on while loops/break/continue unsupported
    if node.orelse:
        raise Exception('else statement on while loop is not supported')
    
    node.test, contr, x, z = setup_internal_values(node.test, scope, contr, x, z)
    x += 1
    num_branches += 1
    contr.add_block((x, z), CommandBlock(
        'scoreboard players test {0} * -1'.format(get_player_and_obj(node.test, scope))
    ))
    contr, x, z = add_pulsegiver_block(contr, x, z)
    x += 1
    contr.add_block((x, z), CommandBlock(
        'scoreboard players test {0} 1 *'.format(get_player_and_obj(node.test, scope))
    ))
    contr, x, z = add_pulsegiver_block(contr, x, z)
    x += 1
    
    # while body branch
    xz = x, z
    x = 0
    z = num_branches - 1
    old_z = z
    
    for stmt in node.body:
        contr, x, z = parse_node(stmt, scope, contr, x, z)
    
    node.test, contr, x, z = setup_internal_values(node.test, scope, contr, x, z, redef=True)
    x += 1
    contr.add_block((x, z), CommandBlock(
        'scoreboard players test {0} 0 0'.format(get_player_and_obj(node.test, scope))
    ))
    contr, x, z = add_pulsegiver_block(contr, x, z, *xz)  # gives control back to while caller
    x += 1
    contr.add_block((x, z), CommandBlock(
        'scoreboard players test {0} * -1'.format(get_player_and_obj(node.test, scope))
    ))
    contr, x, z = add_pulsegiver_block(contr, x, z, wz=old_z)  # gives pulse to own branch
    x += 1
    contr.add_block((x, z), CommandBlock(
        'scoreboard players test {0} 1 *'.format(get_player_and_obj(node.test, scope))
    ))
    contr, x, z = add_pulsegiver_block(contr, x, z, wz=old_z)
    
    x, z = xz
    
    return contr, x, z


@parser(ast.For)
def parse_for_loop(node: ast.For, scope: Scope, contr: Contraption, x: int, z: int) -> Tuple[Contraption, int, int]:
    global num_branches
    
    # 'else' on for loops is unsupported
    if node.orelse:
        raise Exception('else statement on for loop is not supported.')
    
    if not isinstance(node.iter, ast.Name):
        raise Exception('Only names can be iterated over. Try assigning your list to a name.')
    
    node.iter.id = scope.transform(node.iter.id)
    
    if node.iter.id not in listids:
        raise Exception('Cannot iterate over non-list.')
    
    if not isinstance(node.target, ast.Name):
        raise Exception('Only names can be iterator variables.')
    
    node.target.id = scope.transform(node.target.id)
    
    x += 1
    contr.add_block((x, z), CommandBlock('scoreboard players set cntr py2cb_intrnl 0'))
    
    # If the iterator's not empty (has a 0th element), jump to the for body branch
    x += 1
    contr.add_block((x, z), CommandBlock(
        'testfor @e[type=ArmorStand,tag=list,score_py2cb_ids={0},score_py2cb_ids_min={0},score_py2cb_idxs=0,'
        'score_py2cb_idxs_min=0]'.format(listids[node.iter.id])
    ))
    num_branches += 1
    contr, x, z = add_pulsegiver_block(contr, x, z)
    x += 1
    
    # for body branch
    xz = x, z
    x = 0
    z = num_branches - 1
    old_z = z
    x += 1
    contr.add_block((x, z), CommandBlock(
        'scoreboard players operation @e[type=ArmorStand,tag=list,score_py2cb_ids={0},score_py2cb_ids_min={0}] '
        'py2cb_idxs -= cntr py2cb_intrnl'.format(listids[node.iter.id])
    ))
    x += 1
    contr.add_block((x, z), CommandBlock(
        'scoreboard players operation {0} py2cb_var = @e[type=ArmorStand,tag=list,score_py2cb_ids={1},'
        'score_py2cb_ids_min={1},score_py2cb_idxs=0,score_py2cb_idxs_min=0] py2cb_var'
            .format(node.target.id, listids[node.iter.id])
    ))
    x += 1
    contr.add_block((x, z), CommandBlock(
        'scoreboard players operation @e[type=ArmorStand,tag=list,score_py2cb_ids={0},score_py2cb_ids_min={0}] '
        'py2cb_idxs += cntr py2cb_intrnl'.format(listids[node.iter.id])
    ))
    
    for stmt in node.body:
        contr, x, z = parse_node(stmt, scope, contr, x, z)
    
    x += 1
    contr.add_block((x, z), CommandBlock('scoreboard players add cntr py2cb_intrnl 1'))
    x += 1
    contr.add_block((x, z), CommandBlock(
        'scoreboard players operation @e[type=ArmorStand,tag=list,score_py2cb_ids={0},score_py2cb_ids_min={0}] '
        'py2cb_idxs -= cntr py2cb_intrnl'.format(listids[node.iter.id])
    ))
    x += 1
    contr.add_block((x, z), CommandBlock('scoreboard players set forreturn py2cb_intrnl 1'))
    x += 1
    contr.add_block((x, z), CommandBlock(
        'testfor @e[type=ArmorStand,tag=list,score_py2cb_ids={0},score_py2cb_ids_min={0},score_py2cb_idxs=0,'
        'score_py2cb_idxs_min=0]'.format(listids[node.iter.id])
    ))
    x += 1
    contr.add_block((x, z), CommandBlock(
        'scoreboard players set forreturn py2cb_intrnl 0', conditional=True
    ))
    contr, x, z = add_pulsegiver_block(contr, x, z, wz=old_z)
    x += 1
    contr.add_block((x, z), CommandBlock(
        'scoreboard players operation @e[type=ArmorStand,tag=list,score_py2cb_ids={0},score_py2cb_ids_min={0}] '
        'py2cb_idxs += cntr py2cb_intrnl'.format(listids[node.iter.id])
    ))
    x += 1
    contr.add_block((x, z), CommandBlock('scoreboard players test forreturn py2cb_intrnl 1 1'))
    contr, x, z = add_pulsegiver_block(contr, x, z, *xz)
    
    x, z = xz
    
    return contr, x, z


# noinspection PyUnusedLocal
@parser(ast.Break, ast.Continue)
def parse_break_or_continue(node: Union[ast.Break, ast.Continue], scope: Scope, contr: Contraption, x: int, z: int) \
        -> Tuple[Contraption, int, int]:
    raise Exception('break/continue are not supported.')


@parser(ast.Call)
def parse_function_call(node: ast.Call, scope: Scope, contr: Contraption, x: int, z: int) \
        -> Tuple[Contraption, int, int]:
    func_name = get_func_name(node)
    
    # say()
    if func_name == 'say':
        if len(node.keywords):
            raise Exception('say() takes no keyword arguments.')
        
        args = []
        for arg in node.args:
            if isinstance(arg, ast.Str):
                args.append(arg.s)
            elif isinstance(arg, ast.Name) and scope.transform(arg.id) in stringids:
                arg.id = scope.transform(arg.id)
                args.append('@e[type=ArmorStand,tag=string,score_py2cb_var={0},score_py2cb_var_min={0}]'
                            .format(stringids[arg.id]))
            else:
                raise Exception('Only literal strings and names naming strings are supported in say(). '
                                'Use tellraw() for better support.')
        
        x += 1
        contr.add_block((x, z), CommandBlock('say {0}'.format(''.join(args))))
    
    # tell()
    elif func_name == 'tell':
        to = parse_kwargs(node.keywords, [('to', ast.Str('@a'), [ast.Str])], 'tell')['to'].s
        
        # This is also repeated, from say()...
        args = []
        for arg in node.args:
            if isinstance(arg, ast.Str):
                args.append(arg.s)
            elif isinstance(arg, ast.Name) and scope.transform(arg.id) in stringids:
                arg.id = scope.transform(arg.id)
                args.append('@e[type=ArmorStand,tag=string,score_py2cb_var={0},score_py2cb_var_min={0}]'
                            .format(stringids[arg.id]))
            else:
                raise Exception('Only literal strings and names naming strings are supported in say(). '
                                'Use tellraw() for better support.')
        
        x += 1
        contr.add_block((x, z), CommandBlock('tell {0} {1}'.format(to, ''.join(args))))
    
    # tellraw()
    elif func_name == 'tellraw':
        to = parse_kwargs(node.keywords, [('to', ast.Str('@a'), [ast.Str])], 'tellraw')['to'].s
        
        # Each arg is either a raw value, in which case there is no styling applied, or a tuple of raw values
        # + a bitmap style, from the constants above OR'd together
        json_args = []
        for arg in node.args:
            if isinstance(arg, ast.Tuple):
                if not is_bitmap_safe(arg.elts[-1]):
                    raise Exception('Malformed style in tellraw().')
                style = eval(compile(ast.Expression(arg.elts[-1]), '', 'eval'), script.colours_and_styles)
                
                for elem in arg.elts[:-1]:
                    elem, contr, x, z = setup_internal_values(elem, scope, contr, x, z)
                    json_args.append(get_json(elem, scope, style))
            else:
                arg, contr, x, z = setup_internal_values(arg, scope, contr, x, z)
                json_args.append(get_json(arg, scope))
        
        x += 1
        contr.add_block((x, z), CommandBlock(
            'tellraw {0} ["",{1}]'.format(to, ','.join(json_arg for json_arg in json_args))
        ))
    
    # builtin - min()/max()
    elif func_name in ('min', 'max'):
        # We only support min/max of values, not min/max of an iterable
        if node.keywords:
            raise Exception('In Py2CB, {0}() takes no keyword arguments.'.format(func_name))
        
        exprids.add(node)
        
        if len(node.args) == 1:
            node.args[0], contr, x, z = setup_internal_values(node.args[0], scope, contr, x, z)
            x += 1
            contr.add_block((x, z), CommandBlock(
                'scoreboard players operation expr_{0} py2cb_intrnl = {1}'
                    .format(exprids[node], get_player_and_obj(node.args[0], scope))
            ))
            return contr, x, z
        
        for arg in node.args:
            if arg in listids:
                raise Exception('Py2CB\'s {0}() does not support {0} of an iterable.'.format(func_name))
            arg, contr, x, z = setup_internal_values(arg, scope, contr, x, z)
        
        x += 1
        contr.add_block((x, z), CommandBlock(
            'scoreboard players operation expr_{0} py2cb_intrnl = {1}'
                .format(exprids[node], get_player_and_obj(node.args[0], scope))
        ))
        for arg in node.args[1:]:
            x += 1
            contr.add_block((x, z), CommandBlock(
                'scoreboard players operation expr_{0} py2cb_intrnl {2} {1}'
                    .format(exprids[node], get_player_and_obj(arg, scope), {'min': '<', 'max': '>'}[func_name])
            ))
    
    # User-defined function
    elif scope.transform(func_name) in Scope.names_to_scopes:
        func_name = scope.transform(func_name)
        func_scope = Scope.names_to_scopes[func_name]
        
        if not func_call_matches(node, func_scope.node):
            raise Exception('Function call did not match function signature.')
        
        # Set all args to their values / default values in the scope
        # Strings aren't supported yet
        args_dict = get_arg_dict(node, func_scope.node)
        for arg, value in args_dict.items():
            arg = func_scope.transform(arg)
            value, contr, x, z = setup_internal_values(value, scope, contr, x, z)
            
            x += 1
            contr.add_block((x, z), CommandBlock(
                'scoreboard players operation {0} py2cb_var = {1}'.format(arg, get_player_and_obj(value, func_scope))
            ))
        
        # Summon the ArmorStand[tag="return"]
        x += 1
        contr.add_block((x, z), CommandBlock(
            'summon ArmorStand ~2 ~1 ~ {NoGravity:1b,Tags:["return","py2cb"]}'
        ))
        
        # Transfer control to the function
        x += 1
        contr.add_block((x, z), CommandBlock(
            'execute @e[type=ArmorStand,tag=function,score_py2cb_var={0},score_py2cb_var_min={0}] ~ ~-1 ~ {1}'.format(
                func_scope.id, CommandBlock(
                    'setblock ~ ~ ~ minecraft:air', type_=CommandBlock.IMPULSE
                ).get_gen_command(0, 0)
            )
        ))
        
        # Add a space & kill the ArmorStand[tag="return"]
        x += 2  # this thing that looks like a bug isn't
        contr.add_block((x, z), CommandBlock('kill @e[type=ArmorStand,tag=return]'))
    
    # something else
    else:
        raise Exception('Unknown function name "{0}".'.format(func_name))
    
    return contr, x, z


@parser(ast.FunctionDef)
def parse_function_def(node: ast.FunctionDef, scope: Scope, contr: Contraption, x: int, z: int) \
        -> Tuple[Contraption, int, int]:
    global num_branches
    
    node.name = scope.transform(node.name)
    function_scope = Scope(node.name, node, scope)
    
    num_branches += 1
    
    x += 1
    contr.add_block((x, z), CommandBlock(
        'kill @e[type=ArmorStand,tag=function,score_py2cb_var={0},score_py2cb_var_min={0}]'.format(function_scope.id)
    ))
    x += 1
    contr.add_block((x, z), CommandBlock(
        'summon ArmorStand ~{0} ~1 ~{1} {{NoGravity:1b,Tags:["function_noname","py2cb"]}}'
            .format(-x, num_branches - 1)
    ))
    x += 1
    contr.add_block((x, z), CommandBlock(
        'scoreboard players set @e[type=ArmorStand,tag=function_noname] py2cb_var {0}'.format(function_scope.id)
    ))
    x += 1
    contr.add_block((x, z), CommandBlock(
        'entitydata @e[type=ArmorStand,tag=function_noname] {Tags:["function","py2cb"]}'
    ))
    
    xz = x, z
    x = 0
    z = num_branches - 1
    
    for stmt in node.body:
        contr, x, z = parse_node(stmt, function_scope, contr, x, z)
    
    # Return control to the caller, which will have set an armorstand with tag=return at the return location
    x += 1
    contr.add_block((x, z), CommandBlock(
        'execute @e[type=ArmorStand,tag=return] ~ ~-1 ~ {0}'.format(
            CommandBlock('setblock ~ ~ ~ minecraft:air', type_=CommandBlock.IMPULSE, auto=True).get_gen_command(0, 0)
        )
    ))
    
    x, z = xz
    return contr, x, z


@parser(ast.Expr)
def parse_bare_expr(node: ast.Expr, scope: Scope, contr: Contraption, x: int, z: int) -> Tuple[Contraption, int, int]:
    return parse_node(node.value, scope, contr, x, z)


def compile_ast(ast_root: ast.Module) -> Contraption:
    res = Contraption()
    x = z = 0
    res.add_block((x, z), CommandBlock('scoreboard objectives add py2cb_intrnl dummy Py2CB Internal Variables',
                                       type_=CommandBlock.IMPULSE, auto=False))
    x += 1
    res.add_block((x, z), CommandBlock('scoreboard objectives add py2cb_var dummy Py2CB Application Variables'))
    x += 1
    res.add_block((x, z), CommandBlock('scoreboard objectives add py2cb_ids dummy Py2CB IDs'))
    x += 1
    res.add_block((x, z), CommandBlock('scoreboard objectives add py2cb_idxs dummy Py2CB Indexes'))
    x += 1
    res.add_block((x, z), CommandBlock('kill @e[type=ArmorStand,tag=py2cb]'))
    
    for statement in ast_root.body:
        res, x, z = parse_node(statement, Scope.GLOBAL, res, x, z)
    return res


def parse(filename: str) -> Contraption:
    with open(filename) as infile:
        return compile_ast(ast.parse(infile.read(), filename))
