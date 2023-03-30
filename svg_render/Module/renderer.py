#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import xml.etree.ElementTree as ET
from copy import deepcopy

import cv2
import numpy as np
from svgpathtools import parse_path
from tqdm import tqdm

from svg_render.Config.color import COLOR_DICT


class Renderer(object):

    def __init__(self,
                 width=1920,
                 height=1080,
                 free_width=50,
                 render_width=2560,
                 render_height=1440,
                 text_size=1):
        self.render_mode_list = [
            'type', 'semantic', 'selected_semantic', 'custom_semantic',
            'instance'
        ]

        self.width = width
        self.height = height
        self.free_width = free_width
        self.render_width = render_width
        self.render_height = render_height

        self.x_min = float('inf')
        self.y_min = float('inf')
        self.x_max = -float('inf')
        self.y_max = -float('inf')

        self.origin_translate = None
        self.scale = None
        self.post_translate = np.array([self.width / 2.0, self.height / 2.0],
                                       dtype=float)

        self.image = None
        self.resetImage()

        self.image_list = []
        self.text_color = [0, 0, 255]
        self.text_size = text_size
        self.text_line_width = 1

        self.selected_semantic_idx_list = None
        self.custom_semantic_list = None

        self.unknown_dtype_list = []
        return

    def resetImage(self):
        self.image = np.zeros([self.height, self.width, 3], dtype=np.uint8)
        return True

    def resetTransform(self):
        self.x_min = float('inf')
        self.y_min = float('inf')
        self.x_max = -float('inf')
        self.y_max = -float('inf')

        self.origin_translate = None
        self.scale = None

        self.resetImage()
        return True

    def resetWARN(self):
        self.unknown_dtype_list = []
        return True

    def reset(self):
        self.resetTransform()

        self.selected_semantic_idx_list = None
        self.custom_semantic_list = None

        self.resetWARN()
        return True

    def addPoint(self, point):
        self.x_min = min(self.x_min, point[0])
        self.y_min = min(self.y_min, point[1])
        self.x_max = max(self.x_max, point[0])
        self.y_max = max(self.y_max, point[1])
        return True

    def updateViewBox(self, svg_data):
        unknown_dtype_list = []

        for segment, dtype in zip(svg_data['segment_list'],
                                  svg_data['dtype_list']):
            if dtype == 'Line':
                start = [segment.start.real, segment.start.imag]
                end = [segment.end.real, segment.end.imag]

                self.addPoint(start)
                self.addPoint(end)
            elif dtype == 'Arc':
                #  start = [segment.start.real, segment.start.imag]
                #  end = [segment.end.real, segment.end.imag]
                #  rotation = segment.rotation
                #  large_arc = segment.large_arc
                #  sweep = segment.sweep

                pixel_width = int(max(segment.radius.real,
                                      segment.radius.imag)) + 2

                for i in range(pixel_width + 1):
                    point = segment.point(1.0 * i / pixel_width)

                    self.addPoint([point.real, point.imag])
            elif dtype == 'Circle':
                cx = float(segment.attrib['cx'])
                cy = float(segment.attrib['cy'])
                r = float(segment.attrib['r'])

                self.addPoint([cx - r, cy - r])
                self.addPoint([cx + r, cy + r])
            elif dtype == 'Ellipse':
                cx = float(segment.attrib['cx'])
                cy = float(segment.attrib['cy'])
                rx = float(segment.attrib['rx'])
                ry = float(segment.attrib['ry'])

                self.addPoint([cx - rx, cy - ry])
                self.addPoint([cx + rx, cy + ry])
            else:
                if dtype not in unknown_dtype_list:
                    print("[WARN][Renderer::updateViewBox]")
                    print("\t can not solve this segment with type [" + dtype +
                          "]!")
                    unknown_dtype_list.append(dtype)
        return True

    def updateTransform(self, svg_data):
        self.resetTransform()

        self.updateViewBox(svg_data)

        self.origin_translate = np.array([
            -(self.x_min + self.x_max) / 2.0, -(self.y_min + self.y_max) / 2.0
        ],
                                         dtype=float)

        self.scale = min(
            (self.width - 2.0 * self.free_width) / (self.x_max - self.x_min),
            (self.height - 2.0 * self.free_width) / (self.y_max - self.y_min))
        return True

    def getPointInImage(self, point_in_world):
        point_in_image = np.array(deepcopy(point_in_world), dtype=float)
        point_in_image += self.origin_translate
        point_in_image *= self.scale
        point_in_image += self.post_translate
        point_in_image = point_in_image.astype(int)
        return point_in_image

    def getPointInWorld(self, point_in_image):
        point_in_world = np.array(deepcopy(point_in_image), dtype=float)
        point_in_world -= self.post_translate
        point_in_world /= self.scale
        point_in_world -= self.origin_translate
        return point_in_world

    def renderLine(self,
                   line,
                   color,
                   line_width,
                   put_text=None,
                   text_color=[0, 0, 255],
                   text_size=1,
                   text_line_width=1):
        start = [line.start.real, line.start.imag]
        end = [line.end.real, line.end.imag]
        start_in_image = self.getPointInImage(start)
        end_in_image = self.getPointInImage(end)

        cv2.line(self.image, start_in_image, end_in_image, color, line_width)

        if put_text is not None:
            cv2.putText(self.image, put_text,
                        ((start_in_image + end_in_image) / 2.0).astype(int),
                        cv2.FONT_HERSHEY_COMPLEX, text_size, text_color,
                        text_line_width)
        return True

    def renderArc(self,
                  arc,
                  color,
                  line_width,
                  put_text=None,
                  text_color=[0, 0, 255],
                  text_size=1,
                  text_line_width=1):
        #  rotation = arc.rotation
        #  large_arc = arc.large_arc
        #  sweep = arc.sweep

        pixel_width = int(
            max(arc.radius.real * self.scale,
                arc.radius.imag * self.scale)) + 2

        current_point = arc.point(0)
        current_point_in_image = self.getPointInImage(
            [current_point.real, current_point.imag])
        for i in range(1, pixel_width + 1):
            next_point = arc.point(1.0 * i / pixel_width)
            next_point_in_image = self.getPointInImage(
                [next_point.real, next_point.imag])

            cv2.line(self.image, current_point_in_image, next_point_in_image,
                     color, line_width)

            current_point_in_image = next_point_in_image

        if put_text is not None:
            start = [arc.start.real, arc.start.imag]
            end = [arc.end.real, arc.end.imag]
            start_in_image = self.getPointInImage(start)
            end_in_image = self.getPointInImage(end)
            cv2.putText(self.image, put_text,
                        ((start_in_image + end_in_image) / 2.0).astype(int),
                        cv2.FONT_HERSHEY_COMPLEX, text_size, text_color,
                        text_line_width)
        return True

    def renderCircle(self,
                     circle,
                     color,
                     line_width,
                     put_text=None,
                     text_color=[0, 0, 255],
                     text_size=1,
                     text_line_width=1):
        cx = float(circle.attrib['cx'])
        cy = float(circle.attrib['cy'])
        r = float(circle.attrib['r'])

        center_in_image = self.getPointInImage([cx, cy])
        r_in_image = int(r * self.scale)
        cv2.circle(self.image, center_in_image, r_in_image, color, line_width)

        if put_text is not None:
            cv2.putText(self.image, put_text, center_in_image,
                        cv2.FONT_HERSHEY_COMPLEX, text_size, text_color,
                        text_line_width)
        return True

    def renderEllipse(self,
                      ellipse,
                      color,
                      line_width,
                      put_text=None,
                      text_color=[0, 0, 255],
                      text_size=1,
                      text_line_width=1):
        cx = float(ellipse.attrib['cx'])
        cy = float(ellipse.attrib['cy'])
        rx = float(ellipse.attrib['rx'])
        ry = float(ellipse.attrib['ry'])

        center_in_image = self.getPointInImage([cx, cy])
        cv2.ellipse(self.image, center_in_image, [int(rx), int(ry)], 0, 0, 360,
                    color, line_width)

        if put_text is not None:
            cv2.putText(self.image, put_text, center_in_image,
                        cv2.FONT_HERSHEY_COMPLEX, text_size, text_color,
                        text_line_width)
        return True

    def renderSegment(self,
                      segment,
                      dtype,
                      color,
                      line_width,
                      put_text=None,
                      text_color=[0, 0, 255],
                      text_size=1,
                      text_line_width=1):
        if dtype == 'Line':
            return self.renderLine(segment, color, line_width, put_text,
                                   text_color, text_size, text_line_width)
        if dtype == 'Arc':
            return self.renderArc(segment, color, line_width, put_text,
                                  text_color, text_size, text_line_width)
        if dtype == 'Circle':
            return self.renderCircle(segment, color, line_width, put_text,
                                     text_color, text_size, text_line_width)
        if dtype == 'Ellipse':
            return self.renderEllipse(segment, color, line_width, put_text,
                                      text_color, text_size, text_line_width)

        if dtype not in self.unknown_dtype_list:
            self.unknown_dtype_list.append(dtype)
            print("[WARN][Renderer::renderSegment]")
            print("\t can not solve this segment with type [" + dtype + "]!")
        return False

    def updateImageByType(self,
                          svg_data,
                          line_width=1,
                          save_into_list=False,
                          text_color=[0, 0, 255],
                          text_size=1,
                          text_line_width=1):
        for segment, dtype in zip(svg_data['segment_list'],
                                  svg_data['dtype_list']):
            if dtype not in COLOR_DICT.keys():
                color = COLOR_DICT['Others']
            else:
                color = COLOR_DICT[dtype]

            #FIXME: tmp not render text
            put_text = dtype[0]
            put_text = None

            self.renderSegment(segment, dtype, color, line_width, put_text,
                               text_color, text_size, text_line_width)

        if save_into_list:
            self.image_list.append(deepcopy(self.image))
        return True

    def updateImageBySemantic(self,
                              svg_data,
                              line_width=1,
                              save_into_list=False,
                              text_color=[0, 0, 255],
                              text_size=1,
                              text_line_width=1):
        unit_semantic_idx_list = sorted(list(set(
            svg_data['semantic_id_list'])))
        semantic_color_dict = {}
        for unit_semantic_idx in unit_semantic_idx_list:
            semantic_color_dict[str(unit_semantic_idx)] = np.array(
                [
                    np.random.randint(0, 255),
                    np.random.randint(0, 255),
                    np.random.randint(0, 255)
                ],
                dtype=np.uint8).tolist()

        for segment, dtype, semantic_id in zip(svg_data['segment_list'],
                                               svg_data['dtype_list'],
                                               svg_data['semantic_id_list']):
            if semantic_id == 0:
                continue
            self.renderSegment(segment, dtype,
                               semantic_color_dict[str(semantic_id)],
                               line_width, str(semantic_id), text_color,
                               text_size, text_line_width)

        if save_into_list:
            self.image_list.append(deepcopy(self.image))
        return True

    def updateImageBySelectedSemantic(self,
                                      svg_data,
                                      line_width=1,
                                      save_into_list=False,
                                      text_color=[0, 0, 255],
                                      text_size=1,
                                      text_line_width=1):
        if self.selected_semantic_idx_list is None:
            if save_into_list:
                self.image_list.append(deepcopy(self.image))
            return True

        semantic_color_dict = {}
        for unit_semantic_idx in sorted(
                list(set(self.selected_semantic_idx_list))):
            semantic_color_dict[str(unit_semantic_idx)] = np.array(
                [
                    np.random.randint(0, 255),
                    np.random.randint(0, 255),
                    np.random.randint(0, 255)
                ],
                dtype=np.uint8).tolist()

        for segment, dtype, semantic_id in zip(svg_data['segment_list'],
                                               svg_data['dtype_list'],
                                               svg_data['semantic_id_list']):
            if semantic_id not in self.selected_semantic_idx_list:
                continue
            self.renderSegment(segment, dtype,
                               semantic_color_dict[str(semantic_id)],
                               line_width, str(semantic_id), text_color,
                               text_size, text_line_width)

        if save_into_list:
            self.image_list.append(deepcopy(self.image))
        return True

    def updateImageByCustomSemantic(self,
                                    svg_data,
                                    line_width=1,
                                    save_into_list=False,
                                    text_color=[0, 0, 255],
                                    text_size=1,
                                    text_line_width=1):
        if self.custom_semantic_list is None:
            if save_into_list:
                self.image_list.append(deepcopy(self.image))
            return True

        unit_semantic_idx_list = sorted(list(set(self.custom_semantic_list)))

        semantic_color_dict = {}
        for unit_semantic_idx in unit_semantic_idx_list:
            semantic_color_dict[str(unit_semantic_idx)] = np.array(
                [
                    np.random.randint(0, 255),
                    np.random.randint(0, 255),
                    np.random.randint(0, 255)
                ],
                dtype=np.uint8).tolist()
            # FIXME: tmp use white color
            semantic_color_dict[str(unit_semantic_idx)] = np.array(
                [255, 255, 255], dtype=np.uint8).tolist()

        for segment, dtype, semantic_id in zip(svg_data['segment_list'],
                                               svg_data['dtype_list'],
                                               self.custom_semantic_list):
            # FIXME: tmp not render BG
            if semantic_id == 0:
                continue
            self.renderSegment(segment, dtype,
                               semantic_color_dict[str(semantic_id)],
                               line_width, str(semantic_id), text_color,
                               text_size, text_line_width)

        if save_into_list:
            self.image_list.append(deepcopy(self.image))
        return True

    def updateImageByInstance(self,
                              svg_data,
                              line_width=1,
                              save_into_list=False,
                              text_color=[0, 0, 255],
                              text_size=1,
                              text_line_width=1):
        unit_instance_idx_list = sorted(list(set(
            svg_data['instance_id_list'])))
        instance_color_dict = {}
        for unit_instance_idx in unit_instance_idx_list:
            instance_color_dict[str(unit_instance_idx)] = np.array(
                [
                    np.random.randint(0, 255),
                    np.random.randint(0, 255),
                    np.random.randint(0, 255)
                ],
                dtype=np.uint8).tolist()

        for segment, dtype, instance_id in zip(svg_data['segment_list'],
                                               svg_data['dtype_list'],
                                               svg_data['instance_id_list']):
            if instance_id == -1:
                continue
            self.renderSegment(segment, dtype,
                               instance_color_dict[str(instance_id)],
                               line_width, str(instance_id), text_color,
                               text_size, text_line_width)

        if save_into_list:
            self.image_list.append(deepcopy(self.image))
        return True

    def updateImage(self,
                    svg_data,
                    render_mode='type',
                    line_width=1,
                    text_color=[0, 0, 255],
                    text_size=1,
                    text_line_width=1,
                    save_into_list=False):
        self.updateTransform(svg_data)

        if '+' in render_mode:
            sub_render_mode_list = render_mode.split('+')
            for sub_render_mode in sub_render_mode_list:
                self.updateImage(svg_data, sub_render_mode, line_width,
                                 text_color, text_size, text_line_width, True)
            return True

        assert render_mode in self.render_mode_list

        if render_mode == 'type':
            return self.updateImageByType(svg_data, line_width, save_into_list,
                                          text_color, text_size,
                                          text_line_width)
        elif render_mode == 'semantic':
            return self.updateImageBySemantic(svg_data, line_width,
                                              save_into_list, text_color,
                                              text_size, text_line_width)
        elif render_mode == 'selected_semantic':
            return self.updateImageBySelectedSemantic(svg_data, line_width,
                                                      save_into_list,
                                                      text_color, text_size,
                                                      text_line_width)
        elif render_mode == 'custom_semantic':
            return self.updateImageByCustomSemantic(svg_data, line_width,
                                                    save_into_list, text_color,
                                                    text_size, text_line_width)
        elif render_mode == 'instance':
            return self.updateImageByInstance(svg_data, line_width,
                                              save_into_list, text_color,
                                              text_size, text_line_width)
        return True

    def render(self,
               svg_data,
               render_mode='type',
               line_width=1,
               text_color=[0, 0, 255],
               text_size=1,
               text_line_width=1,
               selected_semantic_idx_list=None,
               custom_semantic_list=None):
        self.selected_semantic_idx_list = selected_semantic_idx_list
        self.custom_semantic_list = custom_semantic_list

        self.image_list = []

        self.resetWARN()
        self.updateImage(svg_data, render_mode, line_width, text_color,
                         text_size, text_line_width)
        return True

    def renderFile(self,
                   svg_file_path,
                   render_mode='type',
                   line_width=1,
                   text_color=[0, 0, 255],
                   text_size=1,
                   text_line_width=1,
                   print_progress=False,
                   selected_semantic_idx_list=None,
                   custom_semantic_list=None):
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

        return self.render(svg_data, render_mode, line_width, text_color,
                           text_size, text_line_width,
                           selected_semantic_idx_list, custom_semantic_list)

    def getRenderImage(self):
        if len(self.image_list) == 0:
            return self.image

        render_image = np.hstack(self.image_list)
        return render_image

    def getRenderImageList(self):
        if len(self.image_list) == 0:
            return [self.image]

        return self.image_list

    def show(self, wait_key=0, window_name='[Renderer][image]'):
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, self.render_width, self.render_height)

        render_image = self.getRenderImage()

        cv2.imshow(window_name, render_image)
        cv2.waitKey(wait_key)
        return True
