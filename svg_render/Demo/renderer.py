#!/usr/bin/env python
# -*- coding: utf-8 -*-

from svg_render.Module.renderer import Renderer


def demo():
    svg_file_path = "/home/chli/chLi/FloorPlanCAD/svg/train/0000-0002.svg"

    renderer = Renderer()
    renderer.renderFile(svg_file_path)
    return True
