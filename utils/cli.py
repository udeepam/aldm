"""
Taken from: https://github.com/lmzintgraf/varibad
"""
from distutils.util import strtobool

def boolean_argument(value):
    """Convert a string value to boolean."""
    return bool(strtobool(value))