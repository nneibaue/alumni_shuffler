import zoom_sesh
import pytest
import pandas as pd
import os
import shutil
import time


# This function is used in the TestDirectory context manager below
# and needs to be tested first to make sure the context manager 
# does not break
class TestDirectory():
  '''Context manager to create a temporary directory with fake data.
  
  This is basically a wrapper around `make_fake_data` that ensures
  removal of the directory after the end of the test. This will not
  be applicable for all tests, but some tests do not need persistent
  or specific test directories. Both `make_fake_data` and
  `make_specific_fake_data` are supported

  '''
  def __init__(self, d, fake_data_func, **fake_data_kwargs):

    # Remove directory if it already exists
    if os.path.isdir(d):
      shutil.rmtree(d)

    self._d = d
    self._func = fake_data_func
    self._kwargs = fake_data_kwargs

  def __enter__(self):
    # Make fake data in test directory
    self._func(self._d, **self._kwargs)
    
  def __exit__(self, exc_type, exc_value, exc_traceback):
    # Remove test directory
    shutil.rmtree(self._d)


#========================================================
# FAKE DATA TESTING
#========================================================
def test_make_fake_data():
  d = 'fake_data'

  zoom_sesh.make_fake_data(d, max_people=35)

  assert os.path.isdir(d)
  assert os.listdir(d) == ['alumni.xlsx']

  df = pd.read_excel(os.path.join(d, 'alumni.xlsx'))
  assert list(df.columns) == ['name', 'track', 'year']
  assert len(df) == 35
  # Remove directory after test has finished
  shutil.rmtree(d)


def test_make_fake_data_overwrite_false():
  d = 'fake_data'

  # Make fake data once with overwrite=True (default)
  zoom_sesh.make_fake_data(d)

  with pytest.raises(ValueError) as e:
    # Try making fake data again in the same directory
    zoom_sesh.make_fake_data(d, overwrite=False)  # Should raise exception
    assert f'{d} already exists' in str(e.value)
  
  # Remove directory
  shutil.rmtree(d)


def test_make_specific_fake_data():
  d = 'specific_fake_data'

  # 18 total people
  tracks = {'optics': 5, 'semi': 4, 'polymer': 3, 'sensors': 6}
  years = {'2012': 8, '2013': 7, '2014': 3}
  hair_colors = {'blonde': 11, 'brown': 7}
  zoom_sesh.make_specific_fake_data(d, track=tracks, year=years, hair_color=hair_colors)

  assert os.path.isdir(d)
  assert os.listdir(d) == ['alumni.xlsx']

  df = pd.read_excel(os.path.join(d, 'alumni.xlsx'))
  assert len(df) == 18
  assert list(df.columns) == ['name', 'track', 'year', 'hair_color']
  
  # Test a couple of different categories
  assert len(df[df.track == 'optics'] == 5)
  assert len(df[df.hair_color == 'blonde']) == 11

  shutil.rmtree(d)


#========================================================
# INFRASTRUCTURE TESTING
#========================================================
def test_ZoomSesh_instantiation():
  d = 'instantiation'
  func = zoom_sesh.make_fake_data
  with TestDirectory(d, func):
      z = zoom_sesh.ZoomSesh(d)
      assert isinstance(z, zoom_sesh.ZoomSesh)


def test_breakout_files_exist():
  d = 'breakout_files_exist'
  func = zoom_sesh.make_fake_data
  with TestDirectory(d, func, max_people=6):
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
  d = 'clear_session_dir'
  func = zoom_sesh.make_fake_data
  with TestDirectory(d, func, max_people=6):
    z = zoom_sesh.ZoomSesh(d)
    for _ in range(3):
      z.breakout('track', 2)
    
    time.sleep(1)
    zoom_sesh.clear_session_dir(d)
    
    assert os.listdir(d) == ['alumni.xlsx']


#========================================================
# SPECIFIC BREAKOUT TESTING
#========================================================
def test_single_breakout_equal_tracks_no_extras():
  d = 'equal_tracks_no_extras'
  func = zoom_sesh.make_specific_fake_data
  tracks = {'optics': 5, 'sensors': 5, 'polymers': 5, 'semi': 5}
  with TestDirectory(d, func, track=tracks):
    z = zoom_sesh.ZoomSesh(d)
    b = z.breakout('track', 5)
    assert len(b['extras']) == 0