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
        ns = root.tag[:-3]

        for g in root.iter(ns + 'g'):
            for path in g.iter(ns + 'path'):
                print(path)
                continue
                path_repre = parse_path(path.attrib['d'])
                start = path_repre.point(0)
                end = path_repre.point(1)
                segments.append([start.real, start.imag, end.real, end.imag])
                # starts_ends.append([start.real, start.imag, end.real, end.imag, end.real, end.imag, start.real, start.imag])
                mid = path_repre.point(0.5)
                # length = math.sqrt((start.real - end.real) ** 2 + (start.imag - end.imag) ** 2)
                length = path_repre.length()
                nodes.append([
                    length / width, (mid.real - minx) / width,
                    (mid.imag - miny) / height, 1, 0, 0
                ])
                centers.append([mid.real, mid.imag])
                if 'semantic-id' in path.attrib:
                    semantic_id = int(path.attrib['semantic-id'])
                else:
                    semantic_id = 0
                if 'instance-id' in path.attrib:
                    instance_id = int(path.attrib['instance-id'])
                else:
                    instance_id = -1
            return
            # circle
            for circle in g.iter(ns + 'circle'):
                cx = float(circle.attrib['cx'])
                cy = float(circle.attrib['cy'])
                r = float(circle.attrib['r'])
                segments.append([cx - r, cy, cx + r, cy])
                # starts_ends.append([cx - r, cy, cx + r, cy, cx + r, cy, cx - r, cy])
                nodes.append([
                    r * 2.0 / width, (cx - minx) / width, (cy - miny) / height,
                    0, 1, 0
                ])
                centers.append([cx, cy])
                if 'semantic-id' in circle.attrib:
                    classes.append([int(circle.attrib['semantic-id'])])
                else:
                    classes.append([0])
                if 'instance-id' in circle.attrib:
                    instances.append([int(circle.attrib['instance-id'])])
                else:
                    instances.append([-1])
            # ellipse
            for ellipse in g.iter(ns + 'ellipse'):
                cx = float(ellipse.attrib['cx'])
                cy = float(ellipse.attrib['cy'])
                rx = float(ellipse.attrib['rx'])
                ry = float(ellipse.attrib['ry'])
                segments.append([cx - rx, cy, cx + rx, cy])
                # starts_ends.append([cx - rx, cy, cx + rx, cy, cx + r, cy, cx - r, cy])
                nodes.append([(rx + ry) / width, (cx - minx) / width,
                              (cy - miny) / height, 0, 0, 1])
                centers.append([cx, cy])
                if 'semantic-id' in ellipse.attrib:
                    classes.append([int(ellipse.attrib['semantic-id'])])
                else:
                    classes.append([0])
                if 'instance-id' in ellipse.attrib:
                    instances.append([int(ellipse.attrib['instance-id'])])
                else:
                    instances.append([-1])
        return self.render(svg_data)
