#!usr/bin/env python
# -*- coding: utf-8 -*-
"""
A Py2CB script that demonstrates calling functions.
"""

from py2cb import *


def error(code, num=5):
    tellraw(num, ': ', code)


error(0, 3246)
error(7)
error(num=2, code=1005)


def isequal(ham, eggs):
    if ham == eggs:
        tellraw('Many wows! ', ham, ' == ', eggs)
    else:
        tellraw('I am dissapoint. ', ham, ' != ', eggs)


idxs = [0, 1, 2, 3, 4, 5]
nums = [2, 3, 6, 3, 1, 5]

for idx in idxs:
    isequal(idx, nums[idx])
