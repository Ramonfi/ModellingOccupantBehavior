import os, shutil
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
import locale
import warnings

# Set to German locale to get comma decimal separater
locale.setlocale(locale.LC_NUMERIC, "de_DE.UTF-8")
locale.setlocale(locale.LC_ALL, 'de_DE.UTF-8')

### Beschriftungen
BUID = {'LB':'Leichtbeton','MH':'Massivholz','MW':'Mauerwerk'}
ROOMS = {'WZ':'Wohnzimmer', 'SZ': 'Schlafzimmer', 'B': 'Bad', 'K': 'Küche', 'SWK': '1-Zimmer Appartment'}
APPS = {'N': 'Nord', 'O': 'Ost','S':'Süd'}
AREA = pd.Series({'N': 61.8, 'O': 24.7, 'S': 61.5})
indexnames = {'bui':'Bauwesie', 'app': 'Wohnung', 'room':'Raum', 'value': 'Messwert' }

NAMES = {
    'N': r'Nord', 
    'O': r'Ost',
    'S':r'Süd',
    'LB':r'Leichtbeton',
    'MH':r'Massivholz',
    'MW':r'Mauerwerk',
    'WZ':r'Wohnzimmer',
    'SZ': r'Schlafzimmer',
    'B': r'Bad', 
    'K': r'Küche', 
    'SWK': r'1-Zimmer Appartment',
    'WE': r'Wohneinheit',
    'LogReg': r'Logit',
    'MC': r'Monte Carlo',
    'GEG': r'GEG',
    'Reference': r'Referenz'
 }

SENSORS = pd.Series({
    'HK_IO': 'CALC',
    'Rh':'SENSOR_1min',
    'Rh_amb':'SENSOR_1min',
    'T_amb':'SENSOR_1min',
    'T_amb_g24':'CALC',
    'Tair':'SENSOR_1min',
    'Thk':'SENSOR_1min',
    'Tset':'CALC',
    'g_abs':'CALC',
    'global':'SENSOR_1min',
    'windspeed':'SENSOR_1min',
    'Fenster':'CALC',
    'Fenster Nord [S]':'SENSOR_1min',
    'Fenster West [M]':'SENSOR_1min',
    'Fensteröffnung':'CALC',
    'CO2':'SENSOR_1min',
    'Fenster Ost [L]':'SENSOR_1min',
    'Top':'CALC',
    'Tsk':'SENSOR_1min',
    'Anwesenheit':'CALC',
    'Kaltwasser':'SENSOR_8min',
    'Warmwasser':'SENSOR_8min',
    'Wärmemenge':'SENSOR_5min',
    'Fenster Nord [L]':'SENSOR_1min',
    'Fenster Süd [S]':'SENSOR_1min',
    'Fenster Ost [XL]':'SENSOR_1min',
    'Fenster Süd [M]':'SENSOR_1min',
    'Fenster West [M]-A':'SENSOR_1min',
    'Fenster West [M]-B':'SENSOR_1min'
    }).sort_index()

sensorkind = pd.Series({
    'HK_IO': 0,
    'Rh':1,
    'Rh_amb':1,
    'T_amb':1,
    'T_amb_g24':0,
    'Tair':1,
    'Thk':1,
    'Tset':0,
    'g_abs':0,
    'global':1,
    'windspeed':1,
    'Fenster':0,
    'Fenster Nord [S]':1,
    'Fenster West [M]':1,
    'Fensteröffnung':0,
    'CO2':1,
    'Fenster Ost [L]':1,
    'Top':0,
    'Tsk':1,
    'Anwesenheit':0,
    'Kaltwasser':8,
    'Warmwasser':8,
    'Wärmemenge':5,
    'Fenster Nord [L]':1,
    'Fenster Süd [S]':1,
    'Fenster Ost [XL]':1,
    'Fenster Süd [M]':1,
    'Fenster West [M]-A':1,
    'Fenster West [M]-B':1
    }).sort_index()

sensortypes = pd.Series({
    'HK_IO': 'binär',
    'Rh': 'stetig',
    'Rh_amb': 'stetig',
    'T_amb': 'stetig',
    'T_amb_g24': 'stetig',
    'Tair': 'stetig',
    'Thk': 'stetig',
    'Tset': 'stetig',
    'g_abs': 'stetig',
    'global': 'stetig',
    'windspeed': 'stetig',
    'Fenster': 'kategorisch',
    'Fenster Nord [S]': 'binär',
    'Fenster West [M]': 'binär',
    'Fensteröffnung': 'binär',
    'CO2': 'stetig',
    'Fenster Ost [L]': 'binär',
    'Top': 'stetig',
    'Tsk': 'stetig',
    'Anwesenheit': 'binär',
    'Kaltwasser': 'stetig',
    'Warmwasser': 'stetig',
    'Wärmemenge': 'stetig',
    'Fenster Nord [L]': 'binär',
    'Fenster Süd [S]': 'binär',
    'Fenster Ost [XL]': 'binär',
    'Fenster Süd [M]': 'binär',
    'Fenster West [M]-A': 'binär',
    'Fenster West [M]-B': 'binär'
}).sort_index()


sensor_locations = {'Anwesenheit': ['N_WE', 'O_WE', 'S_WE'],
                    'CO2': ['N_SZ', 'O_SWK', 'S_SZ'],
                    'Fenster': ['N_K', 'N_SZ', 'N_WZ', 'O_SWK', 'S_K', 'S_SZ', 'S_WZ'],
                    'Fenster Nord [L]': ['N_WZ'],
                    'Fenster Nord [S]': ['N_K'],
                    'Fenster Ost [L]': ['N_SZ', 'N_WZ', 'O_SWK', 'S_SZ'],
                    'Fenster Ost [XL]': ['S_WZ'],
                    'Fenster Süd [M]': ['S_WZ'],
                    'Fenster Süd [S]': ['S_K'],
                    'Fenster West [M]': ['N_K', 'N_WZ', 'S_K'],
                    'Fenster West [M]-A': ['S_WZ'],
                    'Fenster West [M]-B': ['S_WZ'],
                    'Fensteröffnung': ['N_K', 'N_SZ', 'N_WZ', 'O_SWK', 'S_K', 'S_SZ', 'S_WZ'],
                    'HK_IO': ['N_B', 'N_SZ', 'N_WZ', 'O_B', 'O_SWK', 'S_B', 'S_SZ', 'S_WZ'],
                    'Kaltwasser': ['N_WE', 'O_WE', 'S_WE'],
                    'Rh': ['N_B',  'N_K',  'N_SZ',  'N_WZ',  'O_B',  'O_SWK',  'S_B',  'S_K',  'S_SZ',  'S_WZ'],
                    'Rh_amb': ['N_B',  'N_K',  'N_SZ',  'N_WZ',  'O_B',  'O_SWK',  'S_B',  'S_K',  'S_SZ',  'S_WZ'],
                    'T_amb': ['N_B',  'N_K',  'N_SZ',  'N_WZ',  'O_B',  'O_SWK',  'S_B',  'S_K',  'S_SZ',  'S_WZ'],
                    'T_amb_g24': ['N_B',  'N_K',  'N_SZ',  'N_WZ',  'O_B',  'O_SWK',  'S_B',  'S_K',  'S_SZ',  'S_WZ'],
                    'Tair': ['N_B',  'N_K',  'N_SZ',  'N_WZ',  'O_B',  'O_SWK',  'S_B',  'S_K',  'S_SZ',  'S_WZ'],
                    'Thk': ['N_B', 'N_SZ', 'N_WZ', 'O_B', 'O_SWK', 'S_B', 'S_SZ', 'S_WZ'],
                    'Top': ['N_SZ', 'N_WZ', 'O_SWK', 'S_SZ', 'S_WZ'],
                    'Tset': ['N_B', 'N_SZ', 'N_WZ', 'O_B', 'O_SWK', 'S_B', 'S_SZ', 'S_WZ'],
                    'Tsk': ['N_SZ', 'N_WZ', 'O_SWK', 'S_SZ', 'S_WZ'],
                    'Warmwasser': ['N_WE', 'O_WE', 'S_WE'],
                    'Wärmemenge': ['N_WE', 'O_WE', 'S_WE'],
                    'g_abs': ['N_B',  'N_K',  'N_SZ',  'N_WZ',  'O_B',  'O_SWK',  'S_B',  'S_K',  'S_SZ', 'S_WZ']}

TexMapper = {
    'CO2': r'$CO_2$', 
    'T_amb': r'$T_{amb}$', 
    'T_amb_g24': r'$T_{amb,g24}$', 
    'Tair': r'$T_{air}$',
    'Rh': 'rH',
    'Rh_amb': r'$rH_{amb}$',
    'g_abs': r'$g_{abs}$',
    }

# Tsk seit 02.09.2021

clrs = sns.color_palette(['#407e9c','#c3553a','#457373',"#F6B53C","#ac80a0","#182d3e"], desat=0.8)
sns.set_palette(clrs, color_codes=True)
HEATMAP = sns.diverging_palette(230, 20, as_cmap=True)
blue = clrs[0]
red = clrs[1]
green = clrs[2]
yellow = clrs[3]
magenta = clrs[4]
dark = clrs[5]
Reds = sns.light_palette(HEATMAP(255), as_cmap=True, n_colors=10)
Blues = sns.light_palette(HEATMAP(0), as_cmap=True, n_colors=10)
Greens = sns.light_palette(green, as_cmap=True, n_colors=10)
CMAP = clrs
CLRS = {
    'LB':blue,
    'MH':yellow,
    'MW':red,
    'N': yellow, 'O': magenta, 'S': green, 
    'K':blue, 'SZ':green, 'WZ': red,'B':magenta, 'SWK':magenta, 
    'Sommer':red, 'Übergang':yellow, 'Winter': blue,
    }

# ======== bbox ==========
bbox_props = dict(boxstyle="round", fc='w', ec='k', alpha=0.9)

params = {
"text.usetex": True,
"text.latex.preamble":  '\n'.join([
    r'\usepackage[utf8]{inputenc}'
    r'\usepackage[T1]{fontenc}'
    r'\usepackage{tgheros}'
    r'\usepackage{sansmath}'
    r'\sansmath'
    r'\usepackage[detect-all,locale=DE]{siunitx}',
    r'\DeclareSIUnit\year{a}'
]),
"font.family": "sans-serif",
"font.serif": [],
"font.sans-serif": [],
"font.monospace": [],
"pgf.texsystem": "pdflatex",

# Use 10pt font in plots to match 10pt font in document
"axes.labelsize": 9,
"font.size": 9,

# Make the legend/label fonts a little smaller
"legend.fontsize": 9,
"xtick.labelsize": 9,
"ytick.labelsize": 9,

"figure.titlesize":   11,     # size of the figure title (``Figure.suptitle()``)
"figure.titleweight": "bold",   # weight of the figure title
"figure.figsize":     (5.866141967621419, 3.6254751188222225),

"legend.frameon":       False,

"grid.linestyle": ":",       # solid

"axes.spines.left":   True,
"axes.spines.bottom": True,
"axes.spines.top":    False,
"axes.spines.right":  False,
#'axes.autolimit_mode':  'round_numbers',
"axes.grid":          False,   # display grid or not
"axes.grid.axis":     "both",    # which axis the grid should apply to
"axes.grid.which":    "both",   # grid lines at {major, minor, both} ticks
"axes.titlelocation": "left",  # alignment of the title: {left, right, center}
"axes.titlesize":     "medium",   # font size of the axes title
"axes.titleweight":   "bold",  # font weight of title

#"lines.linewidth":     .5, 
#"lines.markersize":      .8,
'axes.formatter.use_locale':      True,
"date.autoformatter.year":        r"%Y",
"date.autoformatter.month":       r"%b %y",
"date.autoformatter.day":         r"%d.%b.%y",
"date.autoformatter.hour":        r"%H:%M",
"date.autoformatter.minute":      r"%H:%M",
"date.autoformatter.second":      r"%H:%M:%S",
"date.autoformatter.microsecond": r"%M:%S.%f",
}

def useStyle():
    mpl.rcParams.update(params)
useStyle()

# ======== DIN Formate ==========
def cm(inch):return inch*2.54
def inch(cm):return cm/2.54

DIN = {
    'A6': (inch(10.5), inch(14.8)),
    'A6L': (inch(14.8), inch(10.5)),
    'A5': (inch(14.8), inch(21)),
    'A5L': (inch(21), inch(14.8)),
    'A4': (inch(21), inch(29.7)),
    'A4L': (inch(29.7), inch(21)),
    'A3': (inch(29.7), inch(2*21)),
    'A3L': (inch(2*21), inch(29.7))
    }
    
def uniqueLegend(axesCol, ax, **legend_kwargs):
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

def goldenratio(w=None, h=None):
    # Golden ratio to set aesthetic figure height
    # https://disq.us/p/2940ij3
    if h is None and w is not None:
        return w * ((5**.5 - 1) / 2)
    elif h is not None and w is None:
        return (2*h) / (5**.5 - 1)
    else:
        raise AttributeError('Entweder width oder height übergeben!')

def set_size(width='thesis', fraction=1, aspect=None, subplots=(1, 1)):

    """Set figure dimensions to avoid scaling in LaTeX.

    Parameters
    ----------
    width: float or string
            Document width in points, or string of predined document type
    fraction: float, optional
            Fraction of the width which you wish the figure to occupy
    subplots: array-like, optional
            The number of rows and columns of subplots.
    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    warnings.warn('set_size wird durch size ersetzt....', DeprecationWarning)
    if width == 'thesis':
        width_pt = 423.94608
    elif width == 'beamer':
        width_pt = 307.28987
    else:
        width_pt = width

    # Width of figure (in pts)
    fig_width_pt = width_pt * fraction
    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    # https://disq.us/p/2940ij3
    if aspect == None:
        aspect = (5**.5 - 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * aspect * (subplots[0] / subplots[1])

    return (fig_width_in, fig_height_in)


def size(aspect=None, width='thesis', fraction=1, subplots=(1, 1),give='kwarg', **kwargs):
    """Set figure dimensions to avoid scaling in LaTeX.

    Parameters
    ----------
    aspect: float, optional (default == 'Golden ratio ~ 0.62)
    width: float or string, optional (default = 'thesis')
            Document width in points, or string of predined document type
    fraction: float, optional
            Fraction of the width which you wish the figure to occupy
    subplots: array-like, optional
            The number of rows and columns of subplots.
    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    if 'w' in kwargs:
        width = kwargs['w']
    if 'a' in kwargs:
        aspect = kwargs['a']
    if 'fr' in kwargs:
        fraction = kwargs['fr']

    if width == 'thesis':
        width_pt = 423.94608
    elif width == 'beamer':
        width_pt = 307.28987
    else:
        width_pt = width

    # Width of figure (in pts)
    fig_width_pt = width_pt
    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    # https://disq.us/p/2940ij3
    if aspect == None:
        aspect = (5**.5 - 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * aspect * (subplots[0] / subplots[1])
    if give=='kwarg':
        return {'figsize': (fig_width_in*fraction, fig_height_in)}
    else:
        return (fig_width_in*fraction, fig_height_in)

def toTex(section=None, name=None, fig=None, latex_folder = '.\LaTex', backend='pdf', **savefig_kwargs):
    """
    Exportiere eine figure in mein Latex-Ordner.

    Parameters
    ----------
    filename: string
            Dateiname (ggf. mit Unterordner) OHNE Dateiendung
    fig: mpl.figure
            Zu exportierende Grafik
    export_folder: string or path
            Latex-Grafik-Ordner
    mpl_style: string or path:
            mpl stylesheet mit exportinfos
    kwargs:
            Werden an plt.savefig() weitergebeben
    """
    if section is not None:
        chapters = ['pp', 'desc', 'emp', 'lit', 'model', 'appendix', 'method']
        chapters = {'pp':'preprocessing', 'desc':'Auswertung', 'emp':'Model', 'lit':'Literatur', 'model':'Model', 'appendix':'Anhang', 'method':'Methode'}
        if section not in chapters:
            raise AttributeError(f'Achtung! Unbekanntes Kapitel {section} übergeben. Section sollte in {chapters} sein.')
    if not hasattr(fig, 'savefig'):
        print('exportiere die zuletzt verwendete Figure...')
        fig = plt.gcf()
    fn = os.path.join(latex_folder, 'abb', chapters[section], f'{"_".join(list(filter(None, ["fig", section, name])))}.{backend}')
    _dir = os.path.dirname(fn)
    if not os.path.isdir(_dir): os.makedirs(_dir)
    with mpl.rc_context(params):
        fig.savefig(
            fn, 
            format = backend, 
            backend = backend, 
            bbox_inches='tight', 
            **savefig_kwargs
            )

def exportToLatex(*args, **kwargs):
    """Exportiere eine figure in mein Latex-Ordner.

    Parameters
    ----------
    filename: string
            Dateiname (ggf. mit Unterordner) OHNE Dateiendung
    fig: mpl.figure
            Zu exportierende Grafik
    export_folder: string or path
            Latex-Grafik-Ordner
    mpl_style: string or path:
            mpl stylesheet mit exportinfos
    kwargs:
            Werden an plt.savefig() weitergebeben
    """
    warnings.warn('exportToLatx wird zu toTex', DeprecationWarning)
    toTex(*args, **kwargs)

def cleanBuiAppAxis(ax, which='x', mode='str', sep='-'):
    if which.lower() == 'y':
        if mode.lower() == 'str':
            df = pd.DataFrame([[item._y, *item._text.split(sep)] for item in ax.get_yticklabels()] , columns=['y','bui','app']).apply(lambda x: x.str.strip() if hasattr(x, 'str') else x).replace(BUID)
        elif mode.lower() == 'tuple':
            df = pd.DataFrame([[item._y, *item._text.replace('(','').replace(')','').split(', ')] for item in ax.get_yticklabels()] , columns=['y','bui','app']).apply(lambda x: x.str.strip() if hasattr(x, 'str') else x).replace(BUID)
        ax.set_yticks(list(df.y), list(df.app.rename(NAMES)), minor=True)
        ax.set_yticks(list(df.groupby('bui').mean().y.rename(NAMES)), list(df.groupby('bui').mean().index), minor=False)
        ax.tick_params(width=0, length=15, which='major', axis='y', rotation=90)
        ax.yaxis.remove_overlapping_locs = False
        ax.yaxis.label.set_visible(False)
    elif which.lower() == 'x':
        if mode.lower() == 'str':
            df = pd.DataFrame([[item._x, *item._text.split(sep)] for item in ax.get_xticklabels()], columns=['x','bui','app']).apply(lambda x: x.str.strip() if hasattr(x, 'str') else x).replace(BUID).replace(APPS)
        elif mode.lower() == 'tuple':
            df = pd.DataFrame([[item._x, *item._text.replace('(','').replace(')','').split(', ')] for item in ax.get_xticklabels()], columns=['x','bui','app']).apply(lambda x: x.str.strip() if hasattr(x, 'str') else x).replace(BUID).replace(APPS)
        ax.set_xticks(list(df.x), list(df.app), minor=True)
        ax.set_xticks(list(df.groupby('bui').mean().x.rename(NAMES)), list(df.groupby('bui').mean().index), minor=False)
        ax.tick_params(width=0, length=15, which='major', axis='x', rotation=0)
        ax.xaxis.remove_overlapping_locs = False
        ax.xaxis.label.set_visible(False)

def datemapaxis(ax, axis='y', steps=6, fmt='%H:%M'):
    tick_locs = getattr(ax, f'get_{axis}ticks')()
    lims = ax.dataLim

    step = np.diff(tick_locs)
    
    if np.all(np.isclose(step, step[0])):
        step = np.mean(step)
    else:
        raise ValueError('Tick Abstand ist nicht gleichförmig. Umformung nicht implementiert.')
    
    _min, _max = np.floor(np.min(tick_locs)).astype(int), (np.floor(np.max(tick_locs))+step).astype(int)

    ticks = [x for x in np.linspace(round(getattr(lims, f'{axis}0')) , round(getattr(lims, f'{axis}1')), steps+1)]

    labels = list(pd.date_range(start='00:00', periods=steps+1, freq=f'{24/(steps)}H').strftime('%H:%M'))

    getattr(ax, f'set_{axis}ticks')(ticks, labels, rotation=0)
    getattr(ax, f'set_{axis}label')('Uhrzeit')
    if axis == 'y':
        getattr(ax, f'set_{axis}lim')(ticks[-1], ticks[0])
    else:
        getattr(ax, f'set_{axis}lim')(ticks[0], ticks[-1])