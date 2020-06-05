class HtmlMaker:
  default_style = {
        'table': {
          'border-spacing':'0px',
        },
        'div.horizontal-table': {
          'display': 'inline-table',
          'padding': '1px'
        },
  }
  
  def __init__(self, use_default_style=True):
    self._elements = []
    self.style={}

    if use_default_style:
      self.apply_style(HtmlMaker.default_style.copy())

  
  def _to_css_style(self):
    if not self.style:
      return ''
    css_elements = []
    css_element_template = ('{classname} {{\n  {prop_list}\n  }}\n')
    for classname in self.style:
      inner = self.style[classname]
      prop_list = '\n  '.join(
          [f'{prop_name}: {prop_value};' for prop_name, prop_value in inner.items()])
      css_elements.append(css_element_template.format(
          classname=classname,
          prop_list=prop_list
      ))
    css = ''.join(css_elements)
    return f'<style>\n{css}\n</style>'

  def apply_style(self, style_dict):
    '''Applies a css style dictionary to HtmlMaker object.
  
    Args:
      style_dict: dictionary containing css classes and valid css properties. 

    Note: any css classes declared in `style-dict` will override the default css
      classes in `HtmlMaker.default_style`
  
      Example: 
      >>>default_style = {
      >>>    'div.bluebox': {
      >>>      'border': '2px solid blue',
      >>>      'padding': '10px',
      >>>    },
      >>>    'table': {
      >>>      'border-spacing':'0px',
      >>>    },
      >>>    'div.horizontal': {
      >>>      'display': 'inline-table',
      >>>      'border': '2px solid green',
      >>>      'padding': '10px',
      >>>  }
      >>>}
    '''
    # TODO: Add type checking for `style-dict`. Make sure it's the right format
    # and check to see if it is valid css (somehow)
    self.style.update(style_dict)
        

  def apply_tag(self, enclosing_tag, css_classes=None):
    '''Wraps everything in `enclosing_tag`.'''
    html = self.to_html().data
    self._elements = []
    self.add_html_element(html, enclosing_tag=enclosing_tag, css_classes=css_classes)

        
  def add_pandas_df(self, df, td_class="", th_class="",
                    title=None,
                    include_header=False,
                    include_index=False,
                    enclosing_tag=None,
                    css_classes=None):

    if include_index:
      df = df.reset_index() # Copies the index into a new column

    def get_row(row):
      entries = '' 
      for col in df.columns:
        entries += f'<td class="{td_class}">{row[col]}</td>'
      return f'<tr>{entries}</tr>'

    #body = df.apply(lambda row: '<tr>' + ''.join([f'<td>{row[col]}<td>' for col in df.columns]) + '</tr>', axis=1).values.astype(str)
    body = df.apply(get_row, axis=1)
    body = ''.join(body.values.astype(str))

    if include_header:
      header = ''
      for col in df.columns:
        header += f'<th class="{th_class}">{col}</th>'

      body = header + body

    if not title:
      title = ''
    else:
      title = f'<center><h3>{title}</h3></center>'
    html = (
        f'''
        {title}
        <table>
          {body}
        </table>
        '''
    )

    self.add_html_element(html,
                          enclosing_tag=enclosing_tag,
                          css_classes=css_classes)

  def add_html_maker(self, maker, enclosing_tag=None, css_classes=None):
    '''Adds another html maker to this maker'''
    if not isinstance(maker, HtmlMaker):
      raise TypeError('{maker} is not an instances of HtmlMaker. Try using `add_html_element` instead.')
    html = maker.to_html().data
    self.add_html_element(html, enclosing_tag=enclosing_tag, css_classes=css_classes)

  def add_html_element(self, data,
                       enclosing_tag=None,
                       css_classes=None,
                       insert_at_front=False):
    '''Adds a generic html element to the HtmlMaker

    Args: 
      data: html string to add
      enclosing_tag: optional tag to wrap html
      css_classes: optional list of strings. Which css classes to add to `tag`.
        These muse be the tag identifier without the brackets ('<' and '>')

    Example
    ```
    >>> m = HtmlMaker()
    >>> m.add_html_element("I'm inside a tag!",
                           enclosing_tag='div', 
                           css_classes=["fat-box", "output-area"])
    ```
    '''
    if enclosing_tag is not None:
      if ('<' in enclosing_tag) or ('>' in enclosing_tag):
        raise ValueError('Brackets must not be included. Use tag name by itself.')

      if css_classes is not None:
        assert isinstance(css_classes, list)
        front = f'<{enclosing_tag} class="{" ".join(css_classes)}">'
        back = f'</{enclosing_tag}>'
      
      else:
        front = f'<{enclosing_tag}>'
        back = f'</{enclosing_tag}>'

    else:
      front = ''
      back = ''
    html = f'{front}{data}{back}'

    if insert_at_front:
      self._elements.insert(0, html)
    else:
      self._elements.append(html)

  def to_html(self):
    # Add css
    self._elements.insert(0, self._to_css_style())
    html = (''.join(self._elements))
    return HTML(html)
