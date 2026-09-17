from faceheatmap.config import LayoutMode, PersonId
from faceheatmap.layout import BBox, get_panels
from faceheatmap.tracker import IdentityTracker


def make_tracker():
    panels = get_panels(1000, 500, LayoutMode.SIDE_BY_SIDE)
    return IdentityTracker(panels)


def test_first_frame_assigns_by_panel_position():
    tracker = make_tracker()
    left = BBox(50, 50, 150, 150)
    right = BBox(850, 50, 950, 150)
    assignment = tracker.assign([left, right])
    assert assignment[PersonId.PERSON_1] == left
    assert assignment[PersonId.PERSON_2] == right


def test_first_frame_assigns_regardless_of_detection_order():
    tracker = make_tracker()
    left = BBox(50, 50, 150, 150)
    right = BBox(850, 50, 950, 150)
    assignment = tracker.assign([right, left])
    assert assignment[PersonId.PERSON_1] == left
    assert assignment[PersonId.PERSON_2] == right


def test_identity_persists_across_frames_by_iou_even_if_order_flips():
    tracker = make_tracker()
    left = BBox(50, 50, 150, 150)
    right = BBox(850, 50, 950, 150)
    tracker.assign([left, right])

    # Next frame: detections come back in swapped order but barely moved.
    left2 = BBox(55, 52, 155, 152)
    right2 = BBox(845, 48, 945, 148)
    assignment = tracker.assign([right2, left2])
    assert assignment[PersonId.PERSON_1] == left2
    assert assignment[PersonId.PERSON_2] == right2


def test_single_detection_keeps_prior_identity_via_iou():
    tracker = make_tracker()
    left = BBox(50, 50, 150, 150)
    right = BBox(850, 50, 950, 150)
    tracker.assign([left, right])

    # Person 2 drops out this frame; person 1 barely moved.
    left2 = BBox(52, 51, 152, 151)
    assignment = tracker.assign([left2])
    assert assignment == {PersonId.PERSON_1: left2}


def test_single_detection_with_no_history_falls_back_to_panel():
    tracker = make_tracker()
    right = BBox(850, 50, 950, 150)
    assignment = tracker.assign([right])
    assert assignment == {PersonId.PERSON_2: right}


def test_recovers_after_a_missed_frame():
    tracker = make_tracker()
    left = BBox(50, 50, 150, 150)
    right = BBox(850, 50, 950, 150)
    tracker.assign([left, right])
    tracker.assign([left])  # person 2 missed this frame

    left3 = BBox(53, 53, 153, 153)
    right3 = BBox(847, 47, 947, 147)
    assignment = tracker.assign([left3, right3])
    assert assignment[PersonId.PERSON_1] == left3
    assert assignment[PersonId.PERSON_2] == right3
