#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import cv2
import numpy as np
import xml.etree.ElementTree as ET
from copy import deepcopy
from svgpathtools import parse_path


class Renderer(object):

    def __init__(self):
        return

    def render(self, svg_data):
        return True

    def renderFile(self, svg_file_path):
        assert os.path.exists(svg_file_path)

        tree = ET.parse(svg_file_path)
        root = tree.getroot()
        return True
