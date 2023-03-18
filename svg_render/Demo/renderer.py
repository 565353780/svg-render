#!/usr/bin/env python
# -*- coding: utf-8 -*-

from svg_render.Module.renderer import Renderer


def demo():
    svg_file_path = "/home/chli/chLi/FloorPlanCAD/svg/train/0000-0002.svg"
    width = 4000
    height = 4000
    free_width = 50
    render_width = 2560
    render_height = 1440
    line_width = 1
    text_color = [0, 0, 255]
    text_size = 1
    text_line_width = 1
    print_progress = True

    renderer = Renderer(width, height, free_width, render_width, render_height)
    renderer.renderFile(svg_file_path, line_width, text_color, text_size,
                        text_line_width, print_progress)
    return True
