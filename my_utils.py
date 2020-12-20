# -*- coding: utf-8 -*-
# @Time      :   2020/8/17 12:35
# @Author    :   nicahead@gmail.com
# @File      :   my_utils.py
# @Desc      :
from config import config


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass
    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass
    return False
