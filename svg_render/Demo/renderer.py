#!/usr/bin/env python
# -*- coding: utf-8 -*-

from svg_render.Module.renderer import Renderer


def demo():
    svg_file_path = "/home/chli/chLi/FloorPlanCAD/svg/train/0000-0002.svg"
    render_mode = 'type+semantic+selected_semantic+custom_semantic'
    render_mode = 'type+semantic+selected_semantic'
    width = 4000
    height = 4000
    free_width = 50
    render_width = 2560
    render_height = 1440
    line_width = 3
    text_color = [0, 0, 255]
    text_size = 1
    text_line_width = 1
    print_progress = True
    selected_semantic_idx_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 33, 34]
    custom_semantic_list = None
    wait_key = 0
    window_name = '[Renderer][' + render_mode + ']'

    renderer = Renderer(width, height, free_width, render_width, render_height)
    renderer.renderFile(svg_file_path, render_mode, line_width, text_color,
                        text_size, text_line_width, print_progress,
                        selected_semantic_idx_list, custom_semantic_list)
    renderer.show(wait_key, window_name)
    return True
