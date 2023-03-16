#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import xml.etree.ElementTree as ET
from tqdm import tqdm
from copy import deepcopy

import cv2
import numpy as np
from svgpathtools import Arc, Line, parse_path


class Renderer(object):

    def __init__(self):
        return

    def render(self, svg_data):
        for segment, dtype, semantic_id, instance_id in zip(
                svg_data['segment_list'], svg_data['dtype_list'],
                svg_data['semantic_id_list'], svg_data['instance_id_list']):
            if dtype == 'Line':
                start = [float(segment.start.real), float(segment.start.imag)]
                end = [float(segment.end.real), float(segment.end.imag)]
                print(start)
                print(end)
                exit()
            elif dtype == 'Arc':
                exit()
            elif dtype == 'Circle':
                cx = float(segment.attrib['cx'])
                cy = float(segment.attrib['cy'])
                r = float(segment.attrib['r'])
            elif dtype == 'Ellipse':
                cx = float(ellipse.attrib['cx'])
                cy = float(ellipse.attrib['cy'])
                rx = float(ellipse.attrib['rx'])
                ry = float(ellipse.attrib['ry'])
            else:
                print("[WARN][Renderer::render]")
                print("\t can not solve this segment with type [" + dtype +
                      "]!")
        return True

    def renderFile(self, svg_file_path, print_progress=False):
        assert os.path.exists(svg_file_path)

        tree = ET.parse(svg_file_path)

        root = tree.getroot()
        ns = root.tag[:-3]

        svg_data = {
            'segment_list': [],
            'dtype_list': [],
            'semantic_id_list': [],
            'instance_id_list': [],
        }

        if print_progress:
            print("[INFO][Renderer::renderFile]")
            print("\t start loading data...")
            total_num = 0
            for g in root.iter(ns + 'g'):
                total_num += len(list(g.iter(ns + 'path')))
                total_num += len(list(g.iter(ns + 'circle')))
                total_num += len(list(g.iter(ns + 'ellipse')))
            pbar = tqdm(total=total_num)
        for g in root.iter(ns + 'g'):
            for path in g.iter(ns + 'path'):
                semantic_id = 0
                instance_id = -1
                if 'semantic-id' in path.attrib:
                    semantic_id = int(path.attrib['semantic-id'])
                if 'instance-id' in path.attrib:
                    instance_id = int(path.attrib['instance-id'])

                path_repre = parse_path(path.attrib['d'])
                for segment in path_repre:
                    svg_data['segment_list'].append(segment)
                    svg_data['dtype_list'].append(segment.__class__.__name__)
                    svg_data['semantic_id_list'].append(semantic_id)
                    svg_data['instance_id_list'].append(instance_id)
                if print_progress:
                    pbar.update(1)

            for circle in g.iter(ns + 'circle'):
                semantic_id = 0
                instance_id = -1
                if 'semantic-id' in circle.attrib:
                    semantic_id = int(circle.attrib['semantic-id'])
                if 'instance-id' in circle.attrib:
                    instance_id = int(circle.attrib['instance-id'])

                svg_data['segment_list'].append(circle)
                svg_data['dtype_list'].append('Circle')
                svg_data['semantic_id_list'].append(semantic_id)
                svg_data['instance_id_list'].append(instance_id)
                if print_progress:
                    pbar.update(1)

            for ellipse in g.iter(ns + 'ellipse'):
                semantic_id = 0
                instance_id = -1
                if 'semantic-id' in ellipse.attrib:
                    semantic_id = int(ellipse.attrib['semantic-id'])
                if 'instance-id' in ellipse.attrib:
                    instance_id = int(ellipse.attrib['instance-id'])

                svg_data['segment_list'].append(ellipse)
                svg_data['dtype_list'].append('Ellipse')
                svg_data['semantic_id_list'].append(semantic_id)
                svg_data['instance_id_list'].append(instance_id)
                if print_progress:
                    pbar.update(1)

        if print_progress:
            pbar.close()

        return self.render(svg_data)
