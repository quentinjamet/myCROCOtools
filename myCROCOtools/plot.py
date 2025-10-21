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

#---------------------
# colorbar 
#---------------------
def my_cb(fig, ax, cs, orientation='vertical', label=None, position=[0.91, 0.2, 0.01, 0.6]):
  cb = fig.add_axes([0.91, 0.2, 0.01, 0.6])
  cb = fig.colorbar(cs, ax=ax, orientation=orientation, cax=cb)
  cb.set_label(label, fontsize='x-large')
