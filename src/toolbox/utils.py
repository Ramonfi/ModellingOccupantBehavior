from numpy import isin
import pandas as pd
import logging
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from collections.abc import Iterable
from sklearn import linear_model

# ======== Kompass ==========
KOMPASS = {'n': 'Nord', 'o':'Ost', 's': 'Süd', 'w': 'West'}

# ======== WEEKDAYS ==========
WEEKDAYS = {0:'Montag', 1:'Dienstag', 2:'Mittwoch',3:'Donnerstag',4:'Freitag',5:'Samstag',6:'Sonntag'}

## some Tools...
def setup_logger(name, log_file, level=logging.INFO, printlog=False):
    """To setup as many loggers as you want"""
    formatter = logging.Formatter('%(asctime)s -- %(levelname)s -- %(message)s')

    filehandler = logging.FileHandler(log_file) 
    filehandler.setFormatter(formatter)
    filehandler.encoding = 'utf-8'
   
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if len(logger.handlers) == 0:
        logger.addHandler(filehandler)
        logger.info(f'-----  Initialisiere Logger {name}   -----')
        
    if printlog:
        printhandler = logging.StreamHandler() 
        printhandler.setFormatter(formatter)
        if len(logger.handlers) == 1:
            logger.addHandler(filehandler)

    return logger
# ======== running bar command line print ==========
def running_bar(m, m_max, string = 'loading'):
    m_max = m_max-1
    n = int((m/m_max)*100)
    s = f'{string} <' + (n)*'|' + (100-n) * '-' + f'> {n} %       '
    if m < m_max:
        print(s,end='\r',flush=True)
    if m == m_max:
        print(s,end='\r',flush=True)
        s2 = 'finished!'
        s = s2 + (len(s)-len(s2))*' '
        print(s,end='\r')

def rollingdiff(x):
    return x.iloc[-1] - x.iloc[0]

def getSeason(months, mode='str', übergang=True):
    bins = [0,2,5,8,11,12]
    if übergang:
        labels = {'str': ['Winter','Übergang','Sommer','Übergang','Winter'], 'int': [1,2,3,2,1]}
    else:
        labels = {'str': ['Winter','Frühling','Sommer','Herbst','Winter'], 'int': [1,2,3,4,1]}
    if isinstance(months, int):
        for i,(_min, _max) in enumerate([*zip(bins,bins[1:])]):
            if _min < months <=_max:
                return labels[mode][i]
    elif isinstance(months, pd.Series):
        if pd.api.types.is_datetime64_any_dtype(months):
            months = months.dt.month
        elif pd.api.types.is_numeric_dtype(months):
            pass
    elif isinstance(months, pd.core.indexes.datetimes.DatetimeIndex):
        months = months.to_series().dt.month
    elif isinstance(months, pd.MultiIndex):
        for lvl in range(months.nlevels):
            if hasattr(months.get_level_values(lvl), 'month'):
                months = months.to_frame().iloc[:,lvl].dt.month
                break
    elif hasattr(months, 'month'):
        months = months.month
    elif hasattr(months, 'dt'):
        months = months.dt.month
    else:
        raise ValueError('Konnte keine gültige Monatskodierung finden...')

    return pd.cut(months, bins=bins, labels = labels[mode], ordered=False)

def aggDate(s):
    return s.dt.date

def aggMINMAX(s):
    return s.max() - s.min()

def aggFIRSTLAST(s):
    return s.last() - s.first()

def removeOutliner(df, whis=1.5, col=None, drop_index=False, by=None, group_kws=dict()):
    def _removeOutliner(s, whis=1.5, col=None):  
        if col is not None:
            _s = s[col]
        elif isinstance(s, pd.Series):
            _s = s
        elif isinstance(s, pd.DataFrame):
            if s.shape[1] == 1:
                _s = s.squeeze()
            else:
                raise ValueError('Shape must be 1D')
        
        q1, q3 = _s.quantile([0.25, 0.75])
        iqr = q3 - q1
        iqr_int = (q1 - whis * iqr, q3 + whis * iqr)
        is_outliner = _s.clip(*iqr_int).isin(iqr_int)
        
        print(f'{col}: {is_outliner.mean():.0%} Outliner (IQR={iqr_int[0]:.1f}, {iqr_int[1]:.1f}) entfernt')
        return s[~is_outliner]

    if by is not None:
        kws = dict(group_keys=False)
        kws.update(group_kws)
        return df.groupby(by, **kws).apply(lambda x: _removeOutliner(x, whis=whis, col=col)).reset_index(drop=drop_index)
    else:
        return _removeOutliner(df, col=col, whis=whis)

def highlightSeasons(ax, ymin=None, ymax=None, übergang=True, annot_kws=dict(), fillbetween_kw=dict(), tex=True):
    minmax = [x for x in mpl.dates.num2date(ax.get_xlim())]

    _ylim = [x for x in ax.get_ylim()]

    if ymin is None:    
        ymin = _ylim[0]
    if ymax is None:    
        ymax = _ylim[1]

    s = getSeason(pd.date_range(*[x for x in mpl.dates.num2date(ax.get_xlim())]), übergang=übergang).rename('Season')

    df = s.to_frame().join((s != s.shift()).cumsum().rename('season_index')).rename_axis('Datetime').reset_index()
    tags = df.groupby('season_index').agg({'Datetime': ['median',aggMINMAX], 'Season':'first'}).set_axis(['loc', 'dur', 'text'], axis=1).where(lambda df:df.dur>pd.Timedelta(60, 'days')).dropna()
    df.season_index = df.season_index.apply(lambda x: x%2)

    _fillbetween_kw = {'color':'k', 'alpha':0.1, 'ec':'none'}
    _fillbetween_kw.update(fillbetween_kw)
    ax.fill_between(df.Datetime, y1=df.season_index, **_fillbetween_kw)

    _annot_kws = {'ha':'center', 'va':'bottom'}
    _annot_kws.update(annot_kws)

    for key, item in tags.iterrows():
        if tex:
            plt.annotate(xy=(item['loc'], ymin + (ymax - ymin)*0.5), text=r'\texttt{'+f'{item["text"]}'+r'}', **_annot_kws)
        else:
            plt.annotate(xy=(item['loc'], ymin + (ymax - ymin)*0.5), text=item["text"], **_annot_kws)

    ax.set_xlim(minmax[0], minmax[1])
    ax.set_ylim(_ylim[0], _ylim[1])

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

class AxTransformer:
    def __init__(self, datetime_vals=False):
        self.datetime_vals = datetime_vals
        self.lr = linear_model.LinearRegression()
        return
    
    def process_tick_vals(self, tick_vals):
        if not isinstance(tick_vals, Iterable) or isinstance(tick_vals, str):
            tick_vals = [tick_vals]
            
        if self.datetime_vals == True:
            tick_vals = pd.to_datetime(tick_vals).astype(int).values
            
        tick_vals = np.array(tick_vals)
            
        return tick_vals
    
    def fit(self, ax, axis):
        axis = getattr(ax, f'get_{axis}axis')()
        
        tick_locs = axis.get_ticklocs()
        tick_vals = self.process_tick_vals([label._text for label in axis.get_ticklabels()])
        
        self.lr.fit(tick_vals.reshape(-1, 1), tick_locs)
        return
    
    def transform(self, tick_vals):        
        tick_vals = self.process_tick_vals(tick_vals)
        tick_locs = self.lr.predict(np.array(tick_vals).reshape(-1, 1))
        
        return tick_locs
    
def set_date_ticks(ax, start_date, end_date, axis='y', date_format='%Y-%m-%d', **date_range_kwargs):
    dt_rng = pd.date_range(start_date, end_date, **date_range_kwargs)

    ax_transformer = AxTransformer(datetime_vals=True)
    ax_transformer.fit(ax, axis=axis)
    
    getattr(ax, f'set_{axis}ticks')(ax_transformer.transform(dt_rng))
    getattr(ax, f'set_{axis}ticklabels')(dt_rng.strftime(date_format))

    ax.tick_params(axis=axis, which='both', bottom=True, top=False, labelbottom=True)
    
    return ax

def add_freq(idx, freq=None):
    """Add a frequency attribute to idx, through inference or directly.

    Returns a copy.  If `freq` is None, it is inferred.
    """

    idx = idx.copy()
    if freq is None:
        if idx.freq is None:
            freq = pd.infer_freq(idx)
        else:
            return idx
    idx.freq = pd.tseries.frequencies.to_offset(freq)
    if idx.freq is None:
        raise AttributeError('no discernible frequency found to `idx`.  Specify'
                             ' a frequency string with `freq`.')
    return idx

def transparentcmap(cmap):
    my_cmap = cmap(np.arange(cmap.N))
    my_cmap[:,-1] = np.linspace(0, 1, cmap.N)
    my_cmap = mpl.colors.ListedColormap(my_cmap)
    return my_cmap