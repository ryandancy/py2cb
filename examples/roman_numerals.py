#!usr/bin/env python
# -*- coding: utf-8 -*-
"""
A Py2CB script that generates roman numerals and tellraws them digit by digit, separating numerals with a '---'.
"""

from py2cb import *

numeral_nums = [50, 40, 10, 9, 5, 4, 1]

num = 1
while num <= 50:
    loop = True
    n = num
    res_add = ''
    while loop:
        if n == 0:
            loop = False
            tellraw(('---', BLACK | BOLD | STRIKETHROUGH))
        else:
            do_rest = True
            for i in numeral_nums:
                if do_rest and n >= i:
                    if i == 1:
                        res_add = 'I'
                    if i == 4:
                        res_add = 'I\nV'
                    if i == 5:
                        res_add = 'V'
                    if i == 9:
                        res_add = 'I\nX'
                    if i == 10:
                        res_add = 'X'
                    if i == 40:
                        res_add = 'X\nL'
                    if i == 50:
                        res_add = 'L'
                    n -= i
                    do_rest = False
            tellraw((res_add, BLUE))
    num += 1
