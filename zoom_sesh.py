from numba import jit
import pandas as pd
import numpy as np
import random
import os
from itertools import combinations
from timeit import default_timer as timer
from html_maker import HtmlMaker
import aslogging as logging
import shutil
import json
import openpyxl

COLAB_ROOT = '/content'
ALUMNI_FILE = 'alumni.xlsx'

if os.getcwd() == COLAB_ROOT:  # In Colab
  NAMES_DIR = os.path.join(COLAB_ROOT, 'alumni_shuffler', 'names')
else:  # On local machine
  NAMES_DIR = './names'
  
def clear_session_dir(d):
  '''Removes all files from a session directory except alumni.xlsx'''
  for fname in os.listdir(d):
    if fname != 'alumni.xlsx':
      os.remove(os.path.join(d, fname))

def import_random_names(d):
  name_files = [f for f in os.listdir(d) if f.startswith('yob')]
  df = pd.read_csv(os.path.join(d, name_files[0]))
  names_col = df.columns[0]
  return df[names_col]

def make_fake_data(d, max_people=40, overwrite=True):
  names = import_random_names(NAMES_DIR)[:max_people]
  track_names = ['optics', 'semi', 'polymer', 'sensors']
  years = list(map(str, range(2013, 2020)))
  df=pd.DataFrame()
  df['name'] = names
  df['track'] = [random.choice(track_names) for _ in range(len(names))]
  df['year'] = [random.choice(years) for _ in range(len(names))]

  if os.path.isdir(d):
    if overwrite:
      shutil.rmtree(d)
      os.mkdir(d)
    else:
      raise ValueError(f'{d} already exists! Please set `overwrite` to `False`')
  else:
    os.mkdir(d)

  w = pd.ExcelWriter(f'{d}/alumni.xlsx')
  df.iloc[:max_people].to_excel(w, index=False)
  w.save()
  w.close()

def make_specific_fake_data(d, overwrite=True, **attributes):
  '''Creates a session directory with a very specific alumni.xlsx file.

  Args:
    d: name of session_directory to create
    overwrite: bool. Whether to overwrite a session directory that already exists
    attributes: attributes in the resulting alumni.xlsx file, along with the
      number of students to put in different categories for that attribute. For
      example: `track={'optics': 3, 'semi': 4}, year={'2012': 5, '2013', 2}`.
      If the total number of students from each attribute does not add up to the same
      value, this function will result in an error (like if there are 8 total in
      `track` and 9 total in `year`)
  '''

  if os.path.isdir(d):
    if overwrite:
      shutil.rmtree(d)
      os.mkdir(d)
    else:
      raise ValueError(f'{d} already exists! Please set `overwrite` to `False`')
  else:
    os.mkdir(d)

  if not attributes:
    raise ValueError('At least one attribute must be specified')

  names = import_random_names(NAMES_DIR)
  df = pd.DataFrame()
  num_people = None
  for attr in attributes:
    col = []
    flavors = attributes[attr]
    assert isinstance(flavors, dict)
    for flavor in flavors:
      n = flavors[flavor]
      col += [flavor] * n
    if not num_people:
      num_people = len(col)
      df['name'] = names.iloc[:num_people]
    else:
      if len(col) != num_people:
        raise ValueError('Number of people must be the same for all attributes!') 
    df[attr] = col

  w = pd.ExcelWriter(f'{d}/alumni.xlsx')
  df.to_excel(w, index=False)
  w.save()
  w.close()


  



class ZoomSesh:
  '''Object that helps organize large groups of people during a zoom call.'''

  def __init__(self, session_directory):
    '''Creates a ZoomSesh from student file contained in session_directory
    Args:
      session_directory: folder containing important information about a session. This folder
        *must* contain a file called 'alumni.xlsx' containing the columns 'name', 'year',
        and 'track'.
    '''

    if 'alumni.xlsx' not in os.listdir(session_directory):
      raise FileNotFoundError(f"No alumni file found! Please make sure to include 'alumni.xlsx' in {session_directory}")
      
    alumni_file = os.path.join(session_directory, ALUMNI_FILE)
    self._breakout_file = os.path.join(session_directory, 'breakouts.json')  # To save non-sensitive breakout data
    if not os.path.exists(self._breakout_file):
      # Save blank breakout file
      with open(self._breakout_file, 'w') as f:
        json.dump({}, f)
        
    self._session_directory = session_directory
    self._alumni_data = pd.read_excel(alumni_file) # DataFrame with raw data from alumni file

    # Attributes for this zoom session. These are taken from the column names of the alumni file
    # For now, these will likely just be 'name', 'year' and 'track'
    self.attributes = [col for col in self._alumni_data.columns]

    self._alumni_data = self._create_tracking_cols((self._alumni_data))

    self._alumni_history = []
    self._breakout_history = []

  @staticmethod
  def _create_tracking_cols(df):
    df = df.copy()
    for i in df.index:
      df[f'{i}'] = np.zeros(len(df))
      df[f'{i}_cnsctv'] = np.zeros(len(df))
    return df

  @property
  def alumni(self):
    return self._alumni_data.copy()

  # Core algorithm
  # ====================================================
  def breakout(self, by, group_size, diff=False, autosave=True):
    '''Generates a single breakout group based on the current state.
    
    Args:
      by: string identifier to use for combining alumni
      same: bool saying whether to combine alumni based on similaritis
        (same=True) or differences (same=False).
      group_size: integer. Desired group size
      autosave: bool. Whether or not to save the breakouts to excel automatically

    Examples:

    >>> z = ZoomSesh('file.xlsx')
    --------------------------------------------------

    Do as many breakouts as necessary to ensure that everyone of the same
    track sees each other in groups of 4 to 5

    >>> z.breakout('track', (4, 5), same=True, n=None)
    --------------------------------------------------
    Do 2 breakouts of up to 6 people per group where everyone in each group
    is from a different year.

    >>> z.breakout('year', (0, 6), same=False, n=2)
    --------------------------------------------------

    Returns:
      breakouts: dictionary of the form {'breakout_i': DataFrame}, where i is
        the breakout number.

    '''
    alumni = self._alumni_data
    if diff and by != 'all':
      all_extras, all_groups = self._group_split(by, 'diff', group_size)

    elif by in list(alumni.keys()):
      all_extras = []
      all_groups = []
      vals = alumni[by].unique()

      for val in vals:
        extras, groups = self._group_split(by, val, group_size)
        all_extras.append(extras)
        all_groups.append(groups)

      all_extras = [item for sub_item in all_extras for item in sub_item]
      all_groups = [item for sub_item in all_groups for item in sub_item]

    elif by == 'all':
      all_extras, all_groups = self._group_split(by, 'all', group_size)

    else:
      print("Invalid breakout input")
      return {}

    keys = [f'group{counter}' for counter in range(1,len(all_groups)+1)]
    all_groups = [list(g) for g in all_groups]
    breakout_dict = dict(zip(keys,all_groups))
    breakout_dict['extras'] = all_extras

    self._breakout_history.append(breakout_dict)
    self._alumni_history.append(self._alumni_data.copy())
    if autosave:
      self._save_breakout(len(self._breakout_history))
    return breakout_dict


  # TODO: Fill out this docstring
  def _min_combo(self, alumni, by=None, arg=None, group_size=6):
    filter_combos_for_diversity = False
    if by == 'all': # 'diff' is irrelevant, so we don't have to filter combos
      indices = alumni.index
    else:
      if arg == 'diff':
        filter_combos_for_diversity = True
        indices = alumni.index
      else:
        indices = alumni[alumni[by] == arg].index

    n_attrs = len(self.attributes) 
    counts_cols = [col for col in alumni.columns[n_attrs:] if '_' not in col]
    cnsctv_cols = [col for col in alumni.columns[n_attrs:] if '_' in col]
  
    counts_arr = alumni[counts_cols].values
    cnsctv_arr = alumni[cnsctv_cols].values

    # Stack the two arrays side-by-side, giving a shape (N, N*2) where N is
    # the number of students. 
    alumni_arr = np.concatenate([counts_arr, cnsctv_arr], axis=1) 

    @jit(nopython=True, cache=True)
    def is_combo_diverse(combo, col, threshold):
      num_unique_combo = len(np.unique(col[combo]))

      if num_unique_combo < threshold:
        return False
      return True

    @jit(nopython=True, cache=True)
    def get_sum(arr, combo):
      counts_sum = arr[combo][: , combo]
      cnsctv_sum = arr[combo][:, combo + len(arr)]
      return np.sum(counts_sum) + np.sum(cnsctv_sum)

    
    combos = combinations(indices, group_size)
    combo_arr = np.array(list(combos), dtype='int8')

    if combo_arr.size == 0:
      return 1

    if filter_combos_for_diversity:
      col = alumni[by].values.astype(str)
      threshold = min(group_size, len(np.unique(col)))
      good_combos = np.apply_along_axis(lambda combo: is_combo_diverse(combo, col, threshold), axis=1, arr=combo_arr)
      combo_arr = combo_arr[np.where(good_combos)]

    # Apply `get_sum` to each combo 
    sums = np.apply_along_axis(lambda combo: get_sum(alumni_arr, combo), axis=1, arr=combo_arr)

    min_list = np.where(sums == np.amin(sums))[0]

    return list(combo_arr[random.choice(min_list)])

    

  def _group_split(self, by, arg, group_size):
    '''Creates breakout groups based on similarities or differences of various
    alumni identifiers, such as 'track' or 'year'. 

    Args:
      by: 
      arg:
      min_group_size:
      max_group_size:
    
    Returns:
      groups: a list of tuples containing indices for the different breakout
        groups of size `group_size`
      extras: tuple containing leftover students once the groups are full

    Given an alumni matrix, a subset to select groups from (by & arg), and a group size,
    breaks the subset of alumni into as many groups as possible.
    
    Returns the list of groups and the list of alumni left over as extras.
    '''

    prev_combos = []
    extras = []
    alumni = self._alumni_data
    end_of_split = False

    while(True):
      while_start = timer()

      flat_prev_combos = [item for combo in prev_combos for item in combo]
      current_df = alumni[~alumni.index.isin(flat_prev_combos)]

      if end_of_split:
        extras = list(current_df.index)

      elif by == 'all':
        if len(current_df) < group_size:
          extras = list(current_df.index)
          end_of_split = True
        
      elif arg != 'diff':
        if len(current_df[current_df[by] == arg]) < group_size: #!!!! doesn't work for full diff
          extras = list(current_df[current_df[by] == arg].index)
          end_of_split = True


      if end_of_split:
        mask = [str(i)+"_cnsctv" for i in extras]
        alumni.loc[~alumni.index.isin(extras), mask] = 0
        break

      else:
        combo = self._min_combo(current_df, by=by, arg=arg, group_size=group_size)

        if combo == 1:
          end_of_split = True
          pass
      
        else:
          for i in combo:
            mask = list(map(str, combo))
            mask.remove(str(i))
            mask = mask + [index+"_cnsctv" for index in mask]
            alumni.loc[alumni.index == i, mask] += 1

          mask = [str(i)+"_cnsctv" for i in combo]
          alumni.loc[~alumni.index.isin(combo), mask] = 0 

          prev_combos.append(combo)

      while_stop = timer()

      logging.log(f'Time through while: {while_stop - while_start}')

    return extras, prev_combos

  # Output and display funcs
  # ====================================================
  def _save_breakout(self, i):
    '''Saves breakout groups to Excel and breakout dict as json

    Args: 
      i: breakout number
    '''

    if i > len(self._breakout_history):
      raise ValueError(f"Breakout {i} doesn't exist!")

    b = self._breakout_history[i - 1]

    # Read current breakouts into memory
    with open(self._breakout_file, 'r') as current:
      breakouts = json.load(current)
      
    # Add new breakout and write back to disk. Note that this operation overwrites
    # the current breakout of the same name
    breakouts[f'breakout{i}'] = list(b)
    with open(self._breakout_file, 'w') as f:
      json.dump(breakouts, f, indent=2)
    
    # Save breakout to excel
    excel_fname = os.path.join(self._session_directory, 'breakouts.xlsx')
    if os.path.exists(excel_fname):
      book = openpyxl.load_workbook(excel_fname)
      writer = pd.ExcelWriter(excel_fname, engine='openpyxl', mode='a')
      writer.book = book
      writer.sheets = dict((ws.title, ws) for ws in book.worksheets)
    else:
      writer = pd.ExcelWriter(excel_fname, engine='openpyxl')

    row = (i - 1) * (max([len(group) for group in b.values()]) + 2)
    for n, group in enumerate(b):
      col = n * (1 + len(self.attributes))
      df = self._alumni_data[self.attributes].iloc[b[group]]
      df.index.name = group
      df.to_excel(writer, sheet_name='breakouts', startrow=row, startcol=col)
    
    writer.save()
    writer.close()

  def summary_html(self):
    years = dict(self._alumni_data.year.value_counts())
    tracks = dict(self._alumni_data.track.value_counts())

    def split_alumni():
      alumni = self._alumni_data[['name', 'year', 'track']]
      l = len(alumni)
      if l % 10 == 0:
        N = l // 10
      else:
        N = 1 + (l // 10)
      groups = [alumni[10*n:10*(n + 1)] for n in range(0, N)]
      return groups

    def get_counts_df(attr):
      counts_df =  pd.DataFrame(self._alumni_data[attr].value_counts()).reset_index()
      counts_df.columns = [attr, 'count']
      return counts_df

    year_counts = get_counts_df('year')
    track_counts = get_counts_df('track')

    left = HtmlMaker()
    left.add_html_element(f'<h2>Total Attendees: {len(self._alumni_data)}</h2>')
    left.apply_style({
        'td.summary': {
            'padding': '5px',
            'text-align': 'left',
            'border-bottom': '1px solid #ddd'
        },
        'table': {
            'border': '2px solid black',
        }
    })

    # Shorthand for making tables display inline
    htable = {'enclosing_tag': 'div', 'css_classes': ['horizontal-table']}

    left.add_pandas_df(year_counts, td_class='summary',
                       include_header=True, **htable)
    left.add_pandas_df(track_counts, td_class='summary',
                       include_header=True, **htable)
    right = HtmlMaker()
    for group in split_alumni():
      right.add_pandas_df(group, td_class='summary',
                          include_header=True, **htable)

    summary = HtmlMaker()
    summary.apply_style({
        'div.greenbox': {
            'border': '3px solid green',
            'display': 'inline-block'
        }
    })

    summary.apply_style({
        'span': {
            'display': 'inline-block',
            'padding': '20px',
        },
        'span.left': {
            'float': 'left',
        },
        'span.right': {
            'float': 'right',
        }
    })
    summary.add_html_maker(left, enclosing_tag='span', css_classes=['left'])
    summary.add_html_maker(right, enclosing_tag='span', css_classes=['right'])
    summary.apply_tag('div', css_classes=['greenbox'])
    return summary.to_html()
