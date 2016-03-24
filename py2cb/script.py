#!usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Sequence, Any, Optional, TypeVar, Callable, List
from colorama import Fore, Style

__author__ = 'Copper'


def say(*args: Sequence[Any]) -> None:
    print('[@] ', *args, sep='')


def tell(*args: Sequence[Any], to: Optional[str] = None) -> None:
    tellraw(('@ whispers to you: ', *args, GREY | ITALIC), to=to)


# Flags for tellraw
BLACK = 0x00F
DARK_BLUE = 0x001
DARK_GREEN = 0x002
DARK_AQUA = 0x003
DARK_RED = 0x004
DARK_PURPLE = 0x005
GOLD = DARK_YELLOW = 0x006
GRAY = GREY = 0x007
DARK_GRAY = DARK_GREY = 0x008
BLUE = 0x009
GREEN = 0x00A
AQUA = 0x00B
RED = 0x00C
LIGHT_PURPLE = PURPLE = 0x00D
YELLOW = 0x00E
WHITE = 0x000  # white is the default
BOLD = 0x010
ITALIC = 0x020
UNDERLINED = 0x040
STRIKETHROUGH = 0x080
OBFUSCATED = 0x100

COLOUR_MASK = 0x00F

# For colours with DARK_ variants, the DARK_ version is ANSI's 'normal', and the light/normal version is ANSI's 'bright'
bitmap_to_ansi = {
    BLACK: Fore.BLACK,
    DARK_BLUE: Fore.BLUE,
    DARK_GREEN: Fore.GREEN,
    DARK_AQUA: Fore.CYAN,
    DARK_RED: Fore.RED,
    DARK_PURPLE: Fore.MAGENTA,
    DARK_YELLOW: Fore.YELLOW,  # also GOLD
    GREY: Fore.WHITE + Style.DIM,  # also GRAY
    DARK_GREY: Fore.BLACK + Style.BRIGHT,  # also DARK_GRAY
    BLUE: Fore.BLUE + Style.BRIGHT,
    GREEN: Fore.GREEN + Style.BRIGHT,
    AQUA: Fore.CYAN + Style.BRIGHT,
    RED: Fore.RED + Style.BRIGHT,
    PURPLE: Fore.MAGENTA + Style.BRIGHT,  # also LIGHT_PURPLE
    YELLOW: Fore.YELLOW + Style.BRIGHT,
    WHITE: Fore.WHITE + Style.BRIGHT,
    # None of the non-color formatting flags are *really* supported
    BOLD: '[BOLD] ',
    ITALIC: '[ITALIC] ',
    UNDERLINED: '[UNDERLINED] ',  # There is an ANSI escape for this, but there's no equivalent on Windows
    STRIKETHROUGH: '[STRIKETHROUGH] ',  # Same
    OBFUSCATED: '[OBFUSCATED] '
}

colours_to_mc = {
    BLACK: 'black',
    DARK_BLUE: 'dark_blue',
    DARK_GREEN: 'dark_green',
    DARK_AQUA: 'dark_aqua',
    DARK_RED: 'dark_red',
    DARK_PURPLE: 'dark_purple',
    GOLD: 'gold',  # also DARK_YELLOW
    GRAY: 'gray',  # also GREY
    DARK_GRAY: 'dark_gray',  # also DARK_GREY
    BLUE: 'blue',
    GREEN: 'green',
    AQUA: 'aqua',
    RED: 'red',
    LIGHT_PURPLE: 'light_purple',  # also PURPLE
    YELLOW: 'yellow',
    WHITE: 'white'
}


def tellraw(*args: Sequence[Any], to: Optional[str] = None) -> None:
    """
    Each formatted arg is a tuple (*stuff, flags). Uses ANSI colour codes + Colorama.
    """
    if to is not None:
        print(to, end='')
    
    for arg in args:
        if isinstance(arg, tuple):
            print(''.join(parse_style_bitmap(arg[-1], lambda name, value: bitmap_to_ansi[value])), *arg[:-1],
                  sep='', end='')
        else:
            print(arg, end='')
    print()


T = TypeVar('T')


def parse_style_bitmap(style: int, callback: Callable[[str, int], T]) -> List[T]:
    styles = []
    
    colour_id = style & COLOUR_MASK
    styles.append(callback('color', colour_id))
    
    for name, bitmap in zip(['bold', 'italic', 'underlined', 'strikethrough', 'obfuscated'],
                            [BOLD, ITALIC, UNDERLINED, STRIKETHROUGH, OBFUSCATED]):
        if style & bitmap != 0:
            styles.append(callback(name, bitmap))
    
    return styles
