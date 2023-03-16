#!/usr/bin/env python
# -*- coding: utf-8 -*-

from svg_render.Module.renderer import Renderer


def demo():
    svg_file_path = "/home/chli/chLi/FloorPlanCAD/svg/train/0000-0002.svg"
    width = 1440
    height = 900
    free_width = 50
    line_width = 1
    print_progress = True

    renderer = Renderer(width, height, free_width)
    renderer.renderFile(svg_file_path, line_width, print_progress)
    return True
