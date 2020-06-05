import pandas as pd
import numpy as np
import random
import os
from google.colab import drive, widgets
from IPython.display import display, HTML
from itertools import combinations
drive.mount('/content/gdrive')
NAMES_DIR = '/content/gdrive/My Drive/software_development/alumni_shuffler/names'

def import_names(dir):
  name_files = [f for f in os.listdir(NAMES_DIR) if f.startswith('yob')]
  df = pd.read_csv(os.path.join(dir, name_files[0]))
  return df

def make_fake_data(max_people=40):
  df = import_names(NAMES_DIR)
  track_names = ['optics', 'semi', 'polymer', 'sensors']
  years = list(map(str, range(2013, 2020)))
  df.columns = ['name', 'year', 'track']
  df['track'] = [random.choice(track_names) for _ in range(len(df))]
  #df['track'] = ['polymer' for _ in range(len(df))] <- previously used a polymer only array for testing purposes
  df['year'] = [random.choice(years) for _ in range(len(df))]

  df = df.iloc[:max_people]  
  person_id = list(map(str,np.arange(max_people).tolist()))

  for i in person_id:
    df[i] = np.zeros(len(df))
    df[i+"_cnsctv"] = np.zeros(len(df))

  return df.iloc[:max_people]

class ZoomSesh:
  '''Object that helps organize large groups of people during a zoom call.'''

  def __init__(self, filename=None, max_people=40):
    '''Constructor
    Args:
      filename: location of a file containing the alumni for this session. Columns
        TBD. If `None`, then ZoomSesh will initialize with fake data of len `max_people`.
  
      max_people: int specifying number of people for fake data. Does not do anything
        unless `filename` is None.
  
    '''
    if filename is not None:
      raise NotImplementedError('This feature is not ready yet')
    
    else:
      # For development, testing, debugging, etc.
      self._alumni_history = [make_fake_data(max_people=max_people)]

  @property
  def alumni(self):
    '''Returns 'current' alumni matrix, which is the top matrix in the stack.'''
    return self._alumni_history[0]


  def breakout(self, by, group_size, diff=False, n=None):
    '''Generates a single breakout group based on the current state.
    
    Args:
      by: string identifier to use for combining alumni
      same: bool saying whether to combine alumni based on similaritis
        (same=True) or differences (same=False).
      group_size: tuple specifying range of acceptable group sizes
      n: number of subsequent breakouts. if n=None, then will return the min
        number of breakouts required for everyone to see everyone else according
        to `by` and `same`. 

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
    alumni = self.alumni
    if diff:
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

    breakout_dict = {}
    breakout_dict['extras'] = all_extras
    group_counter = 0

    for group in all_groups:
      group_counter += 1
      breakout_dict[f'group{group_counter}'] = group

    return breakout_dict


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

    if by == 'all' or arg == 'diff':
      indices = alumni.index
    else:
      indices = alumni[alumni[by] == arg].index

    combos = list(combinations(indices,group_size))

    #!!! Current diff is only for year OR track. Full diff (each group member has different year and different track) not implemented yet
    if arg == 'diff':
      temp_combos = []
      for combo in combos:
        vals = alumni.loc[alumni.index.isin(combo),by]
        if len(vals) == len(set(vals)):
          temp_combos.append(combo)
      if len(temp_combos) == 0:
        return 1    
      combos = temp_combos

    sums = []


    for combo in combos:
      temp_sum = 0

      for i in combo:
        twoD_mask = [(col,col+"_cnsctv") for col in list(map(str,combo)) if col != str(i)]
        mask = [col for sub_col in twoD_mask for col in sub_col]
        line = alumni[alumni.index == i][mask]
        temp_sum += np.sum(line.values)

      sums.append(temp_sum)


    sums = np.array(sums)
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
    alumni = self.alumni
    end_of_split = False

    while(True):
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

    return extras, prev_combos


  def summary_html(self):
    years = dict(self.alumni.year.value_counts())
    tracks = dict(self.alumni.track.value_counts())

    #css for table
    style = '''
    table {
      border: 1px solid black;
      width: auto;
      padding:5px;
    }

    .pandas-table {
      display:inline-block;
      border: 10px solid black;
      width: 20%;
      padding:10px;
    }

    th, td, {
      padding: 5px;
      text-align: left;
      border: 1px solid #ddd
    }

    .grid-container {
      display: grid;
      grid-template-columns: 25% 75%;
      background-color: #007030;
      padding: 10px;
    }

    .grid-item {
      background-color: rgba(255, 255, 255);
      border: 1px solid rgba(0, 0, 0, 0.8);
      padding: 10px;
      font-size: 12px;
      text-align: center;
      }

      .left {
        grid-column-start: 1;
        grid-column-end: 2;
      }
      .right {
        grid-column-start: 2
      }


    span.inlineTable {
      display: inline-block;
      vertical-align: text-top;
      padding: 5px;
    }
    '''
    def split_alumni():
      alumni = self.alumni[['name', 'year', 'track']]
      l = len(alumni)
      if l % 10 == 0:
        N = l // 10
      else:
        N = 1 + (l // 10)
      groups = [alumni[10*n:10*(n + 1)].to_html(classes="pandas-table", border=5) for n in range(0, N)]
      return groups

    row = lambda entry: f'<tr>{entry}</tr>'
    row_entries = lambda d: ''.join([
                    row(f'<td>{key}</td><td>{value}</td>') for key, value in d.items()])
    header = lambda attr: row(f'<th>{attr}</th><th>count</th>')

    year_table = f'''<table>
                     <caption>Years</caption>
                     {header("year")}{row_entries(years)}
                     </table>'''
    track_table = f'''<table>
                     <caption>Tracks</caption>
                     {header("track")}{row_entries(tracks)}
                     </table>'''

    html = '''
    <head>
      <style>{style}</style>
    </head>
    <body>
      <div class="grid-container">
        <div class="grid-item left">
          <h2>Total Attendees: {n}</h2>
          <span class="inlineTable">
            {year_table}
          </span>
          <span class="inlineTable">
            {track_table}
          </span>
        </div>
        <div class="grid-item right">
          {alumni}
        </div>
      </div>
    </body>'''
    return HTML(html.format(n=len(self.alumni),
                            style=style,
                            year_table=year_table,
                            track_table=track_table,
                            alumni=''.join(split_alumni())
                            ))
