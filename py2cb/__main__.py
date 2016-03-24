#!usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import py2cb.compiler

__author__ = 'Copper'

if sys.stdout.isatty():
    import colorama as colourama
    colourama.init(autoreset=True)

py2cb.compiler.main()
