import zoom_sesh
import pytest
import pandas as pd
import os
import shutil
import time

# This function is used in the TestDirectory context manager below
# and needs to be tested first to make sure the context manager 
# does not break
def test_make_fake_data_default():
  d = 'test_sessions/fake_data'

  # Make sure that this directory does not already exist, as
  # we are testing its creation. Usually, logic shouldn't be in 
  # tests, but this is a small exception
  try:
    shutil.rmtree(d)
  except FileNotFoundError:
    pass

  zoom_sesh.make_fake_data(d)

  assert os.path.isdir(d)
  assert os.listdir(d) == ['alumni.xlsx']

  # Remove directory after test has finished
  shutil.rmtree(d)


class TestDirectory():
  '''Context manager to create a temporary directory with fake data.
  
  This is basically a wrapper around `make_fake_data` that ensures
  removal of the directory after the end of the test. This will not
  be applicable for all tests, but some tests do not need persistent
  or specific test directories.

  '''
  def __init__(self, d, max_people=40):

    # Remove directory if it already exists
    if os.path.isdir(d):
      shutil.rmtree(d)

    self._d = d
    self._max_people = max_people

  def __enter__(self):
    # Make fake data in test directory
    zoom_sesh.make_fake_data(self._d, self._max_people)

  def __exit__(self, exc_type, exc_value, exc_traceback):
    # Remove test directory
    shutil.rmtree(self._d)


#========================================================
# INFRASTRUCTURE TESTING
#========================================================
def test_ZoomSesh_instantiation():
  d = 'test_sessions/instantiation'
  with TestDirectory(d):
      z = zoom_sesh.ZoomSesh(d)
      assert isinstance(z, zoom_sesh.ZoomSesh)


def test_breakout_files_exit():
  d = 'test_sessions/breakout_files_exist'
  with TestDirectory(d, max_people=6):
    z = zoom_sesh.ZoomSesh(d)

    # Do 5 breakouts by track
    for _ in range(5):
      z.breakout('track', 2)

    time.sleep(1)  # Make sure files finish saving
    expected_files = [ 'alumni.xlsx', 'breakout1.xlsx', 'breakout2.xlsx',
    'breakout3.xlsx', 'breakout4.xlsx', 'breakout5.xlsx', 'breakouts.json']
    actual_files = os.listdir(d)
    for file in expected_files:
      assert file in actual_files


def test_clear_session_dir():
  d = 'test_sessions/clear_session_dir'
  with TestDirectory(d, max_people=6):
    z = zoom_sesh.ZoomSesh(d)
    for _ in range(3):
      z.breakout('track', 2)
    
    time.sleep(1)
    zoom_sesh.clear_session_dir(d)
    
    assert os.listdir(d) == ['alumni.xlsx']


#========================================================
# SPECIFIC BREAKOUT TESTING
#========================================================


def test_breakout_equal_tracks_no_extras():
  d = 'test_sessions/session_5_in_each_track'
  z = zoom_sesh.ZoomSesh(d)
  b = z.breakout('track', 5)
  assert len(b['extras']) == 0