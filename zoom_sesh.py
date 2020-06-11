import pandas as pd
import numpy as np
import random
import os
from itertools import combinations
from timeit import default_timer as timer
from html_maker import HtmlMaker
import aslogging as logging
import shutil

COLAB_ROOT = '/content'

if os.getcwd() == COLAB_ROOT:  # In Colab
  NAMES_DIR = os.path.join(COLAB_ROOT, 'alumni_shuffler', 'names')
else:  # On local machine
  NAMES_DIR = './names'
  
def import_random_names(dir):
  name_files = [f for f in os.listdir(dir) if f.startswith('yob')]
  df = pd.read_csv(os.path.join(dir, name_files[0]))
  return df

def make_fake_data(dir_name, max_people=40, overwrite=True):
  df = import_random_names(NAMES_DIR)
  track_names = ['optics', 'semi', 'polymer', 'sensors']
  years = list(map(str, range(2013, 2020)))
  df.columns = ['name', 'year', 'track']
  df['track'] = [random.choice(track_names) for _ in range(len(df))]
  df['year'] = [random.choice(years) for _ in range(len(df))]

  person_id = list(map(str,np.arange(max_people).tolist()))

  #df = ZoomSesh._create_tracking_cols(df.iloc[:max_people])
  if os.path.isdir(dir_name):
    if overwrite:
      shutil.rmtree(dir_name)
    else:
      raise ValueError(f'{dir_name} already exists! Please set `overwrite` to `False`')
  else:
    os.mkdir(dir_name)

  w = pd.ExcelWriter(f'{dir_name}/alumni.xlsx')
  df.iloc[:max_people].to_excel(w)
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
      
    self._alumni_file = os.path.join(session_directory, 'alumni.xlsx')
    self._session_directory = session_directory
    self._alumni_data = self._create_tracking_cols(pd.read_excel(self._alumni_file)) # DataFrame with raw data from alumni file
    self._alumni_history = []
    self._breakout_history = []

    # Attributes for this zoom session. These are taken from the column names of the alumni file
    # For now, these will likely just be 'name', 'year' and 'track'
    self.attributes = list(self._alumni_data.columns)


  @staticmethod
  def _create_tracking_cols(df):
    df = df.copy()
    for i in df.index:
      df[f'{i}'] = np.zeros(len(df))
      df[f'{i}_cnsctv'] = np.zeros(len(df))
    return df

  def alumni(self):
    return self._alumni.copy()

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
      self.save_breakout(len(self._breakout_history))
    return breakout_dict


  # TODO: Fill out this docstring
  def _min_combo(self, alumni, by=None, arg=None, group_size=6):
    '''Creates a random group that minimizes overlap with alumni in previous breakouts.    

    Args:
      alumni: 
      by:
      arg:
      group_size:

    Returns:
      combos: a tuple of alumni indices for this group
    '''

    logging.log('Begin _min_combo <========================================')
    logging.log(f'size={len(alumni)}, by="{by}", arg="{arg}", group_size={group_size}')

    if by == 'all' or arg == 'diff':
      indices = alumni.index
    else:
      indices = alumni[alumni[by] == arg].index

    combos = list(combinations(indices,group_size))

    #!!! Current diff is only for year OR track. Full diff (each group member has different year and different track) not implemented yet
    if arg == 'diff':
      diff_start = timer()
      temp_combos = []
      for combo in combos:
        vals = alumni.loc[alumni.index.isin(combo),by]
        if len(vals) == len(set(vals)):
          temp_combos.append(combo)
      if len(temp_combos) == 0:
        return 1    
      combos = temp_combos
      diff_end = timer()
      logging.log(f'Diff time: {diff_end-diff_start}')

    twoDmask = [[(str(alumn), str(alumn)+'_cnsctv') for alumn in combos[i]] for i in range(len(combos))]
    masks = [[item for combo in twoDmask[i] for item in combo] for i in range(len(twoDmask))]

    start = timer()
    sums = [np.sum(alumni.loc[alumni.index.isin(combos[i]),masks[i]].values) for i in range(len(combos))] ### HIGHEST COST STEP
    end = timer()

    logging.log(f'Time through highest cost step: {end-start}')
    logging.log('End _min_combo ========================================>')

    min_list = np.where(sums == np.amin(sums))[0]
  
    return combos[random.choice(min_list)]

    

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
  def save_breakout(self, i):
    if i > len(self._breakout_history):
      raise ValueError(f"Breakout {i} doesn't exist!")

    b = self._breakout_history[i - 1]
    
    writer = pd.ExcelWriter(os.path.join(self._session_directory, f'breakout{i}.xlsx'), engine='openpyxl')
    for group in b:
      df = self._alumni_data[self.attributes].iloc[b[group]]
      df.to_excel(writer, sheet_name=group)
    
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
