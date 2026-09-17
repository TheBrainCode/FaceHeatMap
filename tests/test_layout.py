import pytest

from faceheatmap.config import LayoutMode, PersonId
from faceheatmap.layout import BBox, assign_panel, get_panels, infer_layout_from_bboxes


def test_side_by_side_panels_split_width():
    panels = get_panels(1000, 500, LayoutMode.SIDE_BY_SIDE)
    assert panels[PersonId.PERSON_1] == BBox(0, 0, 500, 500)
    assert panels[PersonId.PERSON_2] == BBox(500, 0, 1000, 500)


def test_top_bottom_panels_split_height():
    panels = get_panels(800, 600, LayoutMode.TOP_BOTTOM)
    assert panels[PersonId.PERSON_1] == BBox(0, 0, 800, 300)
    assert panels[PersonId.PERSON_2] == BBox(0, 300, 800, 600)


def test_auto_layout_defaults_to_side_by_side():
    panels = get_panels(1000, 500, LayoutMode.AUTO)
    assert panels == get_panels(1000, 500, LayoutMode.SIDE_BY_SIDE)


def test_infer_layout_side_by_side():
    bbox_a = BBox(0, 0, 100, 100)
    bbox_b = BBox(800, 20, 900, 120)
    assert infer_layout_from_bboxes(bbox_a, bbox_b) is LayoutMode.SIDE_BY_SIDE


def test_infer_layout_top_bottom():
    bbox_a = BBox(0, 0, 100, 100)
    bbox_b = BBox(20, 700, 120, 800)
    assert infer_layout_from_bboxes(bbox_a, bbox_b) is LayoutMode.TOP_BOTTOM


def test_assign_panel_left_and_right():
    panels = get_panels(1000, 500, LayoutMode.SIDE_BY_SIDE)
    left_face = BBox(100, 100, 200, 200)
    right_face = BBox(700, 100, 800, 200)
    assert assign_panel(left_face, panels) is PersonId.PERSON_1
    assert assign_panel(right_face, panels) is PersonId.PERSON_2


def test_assign_panel_falls_back_to_closest_when_outside_both():
    panels = get_panels(1000, 500, LayoutMode.SIDE_BY_SIDE)
    slightly_off = BBox(-50, 100, 50, 200)  # center at x=0, still within panel 1 bounds actually
    assert assign_panel(slightly_off, panels) is PersonId.PERSON_1

    out_of_frame = BBox(-200, -200, -100, -100)  # center at (-150,-150), outside both panels
    assert assign_panel(out_of_frame, panels) is PersonId.PERSON_1
