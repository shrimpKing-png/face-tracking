# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 01:16:25 2025
Last Update: 25JUNE2025
@author: GPAULL
"""

import os
import tkinter as tk
from tkinter import filedialog


def filebrowser(select_directory=False):
    """
    Allows you to browse to a file or directory and return its absolute path as a string

    Parameters
    ----------
    select_directory : bool, optional
        If True, opens directory selection dialog. If False, opens file selection dialog.
        Default is False (file selection).

    Returns
    -------
    str
        File or directory relative path. Returns empty string if no selection made.
    """

    core = tk.Tk()
    core.withdraw()

    if select_directory:
        filepath = filedialog.askdirectory()
    else:
        filepath = filedialog.askopenfilename()

    core.destroy()

    if filepath:
        return os.path.relpath(filepath)
    else:
        return ""
