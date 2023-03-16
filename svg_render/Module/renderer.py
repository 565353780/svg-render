#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import xml.etree.ElementTree as ET

import cv2
import numpy as np
from svgpathtools import parse_path
from tqdm import tqdm
from copy import deepcopy


class Renderer(object):

    def __init__(self, width=1440, height=900, free_width=50):
        self.width = width
        self.height = height
        self.free_width = free_width

        self.origin_translate = None
        self.scale = None
        self.post_translate = np.array([self.width / 2.0, self.height / 2.0],
                                       dtype=float)

        self.image = None
        return

    def updateTransform(self, view_box_start, view_box_size):
        self.origin_translate = np.array([
            -view_box_start[0] - view_box_size[0] / 2.0,
            -view_box_start[1] - view_box_size[1] / 2.0
        ],
                                         dtype=float)

        self.scale = min(
            (self.width - 2.0 * self.free_width) / view_box_size[0],
            (self.height - 2.0 * self.free_width) / view_box_size[1])
        return True

    def getPointInImage(self, point_in_world):
        point_in_image = np.array(deepcopy(point_in_world), dtype=float)
        point_in_image += self.origin_translate
        point_in_image *= self.scale
        point_in_image += self.post_translate
        point_in_image = point_in_image.astype(np.uint8)
        return point_in_image

    def getPointInWorld(self, point_in_image):
        point_in_world = np.array(deepcopy(point_in_image), dtype=float)
        point_in_world -= self.post_translate
        point_in_world /= self.scale
        point_in_world -= self.origin_translate
        return point_in_world

    def getImage(self, svg_data):
        image = np.zeros([width, height, 3], dtype=np.uint8)

        min_x, min_y, view_width, view_height = svg_data['view_box']

        self.updateTransform([min_x, min_y], [view_width, view_height])

        for segment, dtype, semantic_id, instance_id in zip(
                svg_data['segment_list'], svg_data['dtype_list'],
                svg_data['semantic_id_list'], svg_data['instance_id_list']):
            if dtype == 'Line':
                start = [float(segment.start.real), float(segment.start.imag)]
                end = [float(segment.end.real), float(segment.end.imag)]
                print("====Line====")
                print(start)
                print(end)
            elif dtype == 'Arc':
                start = [float(segment.start.real), float(segment.start.imag)]
                end = [float(segment.end.real), float(segment.end.imag)]
                radius = [
                    float(segment.radius.real),
                    float(segment.radius.imag)
                ]
                rotation = float(segment.rotation)
                large_arc = segment.large_arc
                sweep = segment.sweep
                print("====Arc====")
                print(start)
                print(end)
                print(radius)
                print(rotation)
                print(large_arc)
                print(sweep)
                print("iter points ...")
                for i in range(11):
                    print(segment.point(1.0 * i / 10.0))
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
        return image

    def render(self, svg_data):
        image = self.getImage(svg_data)

        return True

    def renderFile(self, svg_file_path, print_progress=False):
        assert os.path.exists(svg_file_path)

        svg_data = {
            'view_box': [],
            'segment_list': [],
            'dtype_list': [],
            'semantic_id_list': [],
            'instance_id_list': [],
        }

        tree = ET.parse(svg_file_path)

        root = tree.getroot()
        ns = root.tag[:-3]

        svg_data['view_box'] = [
            int(float(x)) for x in root.attrib['viewBox'].split(' ')
        ]

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
