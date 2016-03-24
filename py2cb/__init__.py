#!usr/bin/env python
# -*- coding: utf-8 -*-

from py2cb.script import say, tell, tellraw, BLACK, DARK_BLUE, DARK_GREEN, DARK_AQUA, DARK_RED, DARK_PURPLE, \
    DARK_YELLOW, GOLD, GREY, GRAY, DARK_GREY, DARK_GRAY, BLUE, GREEN, AQUA, RED, LIGHT_PURPLE, PURPLE, YELLOW, WHITE, \
    BOLD, ITALIC, UNDERLINED, STRIKETHROUGH, OBFUSCATED

__author__ = 'Copper'

if __name__ == '__main__':
    import sys
    if sys.stdout.isatty():
        import colorama as colourama
        colourama.init(autoreset=True)
    
    import py2cb.compiler
    py2cb.compiler.main()
