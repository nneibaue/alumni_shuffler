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

def test_state_array_book_keeping():
  d = 'test_sessions/session_book_keeping'
  z = zoom_sesh.ZoomSesh(d)

  #Test1
  b1 = z.breakout('all',3)
  extras_line1 = z.alumni[z.alumni.index == b1['extras'][0]].values[0][3:]
  test = extras_line1 == 0.0
  test1 = not False in test

  #Test2
  test2 = True
  groups = ['group1','group2','group3']
  for group in groups:
    for index in b1[group]:
      line = z.alumni.loc[z.alumni.index == index,z.alumni.columns[3:]]
      ones_mask_2d = [(str(i),str(i)+'_cnsctv') for i in b1[group] if i != index]
      ones_mask = [item for sub_item in ones_mask_2d for item in sub_item]
      ones = line[ones_mask].values[0] == 1.0
      zeros_mask = [col for col in list(line.columns) if col not in ones_mask]
      zeros = line[zeros_mask].values[0] == 0.0
      if (False in zeros or False in ones):
        test2 = False

  #Test3
  test3 = True
  b2 = z.breakout('all',3)
  test_alum = b1['group1'][0]
  if test_alum == b2['extras'][0]:
    test_alum = b1['group1'][1]
  for key in b2.keys():
    if test_alum in b2[key]:
      b1_group_members = [member for member in b1['group1'] if member != test_alum]
      b2_group_members = [member for member in b2[key] if member != test_alum]
      break
  for member in b1_group_members:
    if not member in b2_group_members:
      vals = z.alumni.loc[z.alumni.index == test_alum,[str(member),str(member)+'_cnsctv']].values[0]
      if not(vals[0]==1.0 and vals[1]==0.0):
        test3 = False
      break
    
  #Test4
  test4 = True
  if b1['extras'][0] == b2['extras'][0]:
    print('extras equalled extras')
    test4 = False
  else:
    for key in b1.keys():
      if b2['extras'][0] in b1[key]:
        extras_prev_group_members = [member for member in b1[key] if member != b2['extras'][0]]
        break
    line = z.alumni[z.alumni.index == b2['extras'][0]]
    for member in extras_prev_group_members:
      if not(line[str(member)].values[0] == 1.0 and line[str(member)+'_cnsctv'].values[0] == 0.0):
        test4 = False
      member_line = z.alumni[z.alumni.index == member]
      if not(member_line[str(b2['extras'][0])].values[0] == 1.0 and member_line[str(b2['extras'][0])+'_cnsctv'].values[0] == 0.0):
        test4 = False

  assert test1 == test2 == test3 == test4 == True

      

