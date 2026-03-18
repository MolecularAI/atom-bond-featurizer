"""Initialization."""

import os

import bonafide

# Ensure that the correct bonafide and tests directories are used
print()
print(f"BONAFIDE directory:  {os.path.dirname(bonafide.__file__)}")
print(f"Tests directory:     {os.path.dirname(__file__)}")
print()
