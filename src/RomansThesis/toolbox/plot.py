import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import os

def uniqueLegend(axesCol, ax, **legend_kwargs):
    DeprecationWarning('Please Use utils.uniqueLegend!')
    axes = []
    for item in axesCol:
        if isinstance(item, np.ndarray): axes.extend(list(item))
        elif isinstance(item, list): axes.extend(item)
        else: axes.append(item)
    dummie = {lab: han for _ax in axes for han, lab in zip(*_ax.get_legend_handles_labels())}
    for _ax in axes:
        try:
            _ax.get_legend().remove()
        except AttributeError:
            pass
    ax.legend([*dummie.values()], [*dummie], **legend_kwargs)

def highlightSeasons(ax, ymin=None, ymax=None, annot_kws=dict()):
    DeprecationWarning('Please Use utils.highlightSeasons!')
    minmax = [x for x in mpl.dates.num2date(ax.get_xlim())]
    _ylim = [x for x in ax.get_ylim()]
    if ymin == None:    ymin = _ylim[0]
    if ymax == None:    ymax = _ylim[1]
    _seasons = [
        {'Winter': pd.Series(ymax, pd.date_range('2020-12-01', '2021-03-01', tz='Europe/Berlin', freq='D'))},
        {'Übergang': pd.Series(ymin, pd.date_range('2021-03-01', '2021-06-01', tz='Europe/Berlin', freq='D'))},
        {'Sommer': pd.Series(ymax, pd.date_range('2021-06-01', '2021-09-01', tz='Europe/Berlin', freq='D'))},
        {'Übergang': pd.Series(ymin, pd.date_range('2021-09-01', '2021-12-01', tz='Europe/Berlin', freq='D'))},
        {'Winter': pd.Series(ymax, pd.date_range('2021-12-01', '2022-03-01', tz='Europe/Berlin', freq='D'))},
        {'Übergang': pd.Series(ymin, pd.date_range('2022-03-01', '2022-06-01', tz='Europe/Berlin', freq='D'))},
        {'Sommer': pd.Series(ymax, pd.date_range('2022-06-01', '2022-09-01', tz='Europe/Berlin', freq='D'))},
        {'Übergang': pd.Series(ymin, pd.date_range('2022-09-01', '2022-12-01', tz='Europe/Berlin', freq='D'))},
        {'Winter': pd.Series(ymax, pd.date_range('2022-12-01', '2023-03-01', tz='Europe/Berlin', freq='D'))}
        ]
    df = pd.concat([_item for item in _seasons for key, _item in item.items()])
    df = df.where(((df.index > minmax[0]) & (df.index < minmax[1])))
    ax.fill_between(df.index, y1=df, color='k', alpha=0.1, ec='none')
    _annot_kws = {'ha':'center', 'va':'bottom'}
    _annot_kws.update(annot_kws)
    for xy, text in {(item.index.mean(), ymin+0.5): key for s in _seasons for key, item in s.items()}.items():
        ax.annotate(text, xy, **_annot_kws)
    ax.set_xlim(minmax[0], minmax[1])
    ax.set_ylim(_ylim[0], _ylim[1])

def sync_xaxis(axs):
    _min, _max = tuple(map(list, zip(*[list(a.get_xlim()) for a in axs])))
    _min= min(_min)
    _max= max(_max)
    for a in axs:
        a.set_xlim(_min, _max)
        
def sync_yaxis(axs):
    _min, _max = tuple(map(list, zip(*[list(a.get_ylim()) for a in axs])))
    _min= min(_min)
    _max= max(_max)
    for a in axs:
        a.set_ylim(_min, _max)

class Handler(object):
    def __init__(self, handles:tuple):
        self.h1 = handles[0]
        self.h2 = handles[1]

    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        x0, y0 = handlebox.xdescent, handlebox.ydescent
        width, height = handlebox.width, handlebox.height
        width_box = 0.75*width
        x1 = x0 * 0.9
        y1 = y0 * 3
        patch = plt.Rectangle([x1, y1], width_box, 1.5*height, facecolor=self.h1._facecolor,
                                   edgecolor=self.h1._edgecolor, transform=handlebox.get_transform(), alpha=self.h1._alpha)

        patch2 = plt.Rectangle([x1-width_box, y1], width_box, 1.5*height, facecolor=self.h2._facecolor,
                                   edgecolor=self.h2._edgecolor, transform=handlebox.get_transform(), alpha=self.h2._alpha)
        handlebox.add_artist(patch)
        handlebox.add_artist(patch2)
        return patch

# ======== despine ==========
def despine(ax, which='tr'):
    '''
    Entfernt die Axen von einem Graphen.

    Args:
    ----------
    ax <matplotlib.Axes>
    which <str> {'all', 'tr': 'Top + Right', 'tlr': 'Top + Left + Right'} or <list> {'left', 'right', 'top', 'bottom'}: 
    '''
    if isinstance(which, str):
        which = {'all' : ['left', 'right', 'top', 'bottom'], 'tr' : ['top', 'right'], 'tlr' : ['top', 'right', 'left']}[which]
    for item in which:
        ax.spines[item].set_visible(False)