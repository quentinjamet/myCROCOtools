#!/usr/bin/env python

import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText

#------------------
# title
#------------------
def my_tit(ax, tit='toto', loc='upper right'):
  '''
  Define title of figures with patch style.
  '''

  at = AnchoredText(str(r"%s" % tit), prop=dict(size=15), frameon=True, loc=loc)
  at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
  ax.add_artist(at)
