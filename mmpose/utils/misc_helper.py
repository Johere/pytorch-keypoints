import re


def is_number(num):
    pattern = re.compile(r'^[-+]?[-0-9]\d*\.\d*|[-+]?\.?[0-9]\d*$')
    res = pattern.match(num)
    if res:
        return True
    return False


def try_decode(val):
    """int, float, or str"""
    if val.isdigit():
        return int(val)
    if is_number(val):
        return float(val)
    if isinstance(val, str) and val.lower() in ['true', 'false']:
        return val.lower() == 'true'
    return val