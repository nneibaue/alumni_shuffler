import zoom_sesh
import pytest
import pandas as pd



def test_ZoomSesh_instantiation():
  z = zoom_sesh.ZoomSesh('test_sessions')
  assert isinstance(z, zoom_sesh.ZoomSesh)


def test_equal_tracks_no_extras():
  d = 'test_sessions/session_5_in_each_track'
  z = zoom_sesh.ZoomSesh(d)
  b = z.breakout('track', 5)
  assert len(b['extras']) == 0
