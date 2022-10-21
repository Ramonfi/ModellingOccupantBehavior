
import pandas as pd
import numpy as np

from geopy import geocoders
from meteostat import Stations, Hourly
from timezonefinder import TimezoneFinder

import math
import datetime as dt
import matplotlib as mpl
import matplotlib.pyplot as plt

import seaborn as sns
from .. import style

def transparentcmap(cmap):
    my_cmap = cmap(np.arange(cmap.N))
    my_cmap[:,-1] = np.linspace(0, 1, cmap.N)
    my_cmap = mpl.colors.ListedColormap(my_cmap)
    return my_cmap

def rgb_white2alpha(rgb, ensure_increasing=False):
    """
    Convert a set of RGB colors to RGBA with maximum transparency.
    
    The transparency is maximised for each color individually, assuming
    that the background is white.
    
    Parameters
    ----------
    rgb : array_like shaped (N, 3)
        Original colors.
    ensure_increasing : bool, default=False
        Ensure that alpha values are strictly increasing.
    
    Returns
    -------
    rgba : numpy.ndarray shaped (N, 4)
        Colors with maximum possible transparency, assuming a white
        background.
    """
    # The most transparent alpha we can use is given by the min of RGB
    # Convert it from saturation to opacity
    alpha = 1. - np.min(rgb, axis=1)
    if ensure_increasing:
        # Let's also ensure the alpha value is monotonically increasing
        a_max = alpha[0]
        for i, a in enumerate(alpha):
            alpha[i] = a_max = np.maximum(a, a_max)
    alpha = np.expand_dims(alpha, -1)
    # Rescale colors to discount the white that will show through from transparency
    rgb = (rgb + alpha - 1) / alpha
    # Concatenate our alpha channel
    return np.concatenate((rgb, alpha), axis=1)
    
def cmap_white2alpha(cmap, ensure_increasing=False, register=False):
    """
    Convert colormap to have the most transparency possible, assuming white background.
    
    Parameters
    ----------
    cmap : str or mpl.colors.Colormap
        Name of builtin (or registered) colormap.
    ensure_increasing : bool, default=False
        Ensure that alpha values are strictly increasing.
    register : bool, default=True
        Whether to register the new colormap.

    Returns
    -------
    cmap : matplotlib.colors.ListedColormap
        Colormap with alpha set as low as possible.
    """
    # Fetch the cmap callable
    if isinstance(cmap, mpl.colors.Colormap):
        pass
    elif isinstance(cmap, str):
        cmap = plt.get_cmap(cmap)
    # Get the colors out from the colormap LUT
    rgb = cmap(np.arange(cmap.N))[:, :3]  # N-by-3
    # Convert white to alpha
    rgba = rgb_white2alpha(rgb, ensure_increasing=ensure_increasing)
    # Create a new Colormap object
    cmap_alpha = mpl.colors.ListedColormap(rgba, name=cmap.name + "_alpha")
    if register:
        mpl.cm.register_cmap(name=cmap.name + "_alpha", cmap=cmap_alpha)
    return cmap_alpha

def cmap_linear_alpha(cmap, alpha_start=0, alpha_end=1, background="w", register=False):
    """
    Add linearly increasing alpha to a colormap.

    Parameters
    ----------
    name : str
        Name of builtin (or registered) colormap.
    alpha_start : float, default=0
        Initial alpha value.
    alpha_end : float, default=1
        Final alpha value.
    background : str or tuple or None, default="w"
        If this is set, the colours are adjusted to correct for the specified
        background color as much as possible. Default is ``"w"``, which corrects
        the colors for a white background given the new alpha values.
        Set to ``None`` to disable.
    register : bool, default=True
        Whether to register the new colormap.

    Returns
    -------
    cmap : matplotlib.colors.ListedColormap
        Colormap with alpha linearly increasing.
    """
    # Fetch the cmap callable
    if isinstance(cmap, mpl.colors.Colormap):
        pass
    elif isinstance(cmap, str):
        cmap = plt.get_cmap(cmap)
    # Get the colors out from the colormap LUT
    rgb = cmap(np.arange(cmap.N))[:, :3]  # N-by-3
    # Create linearly increasing alpha channel
    alpha = np.linspace(alpha_start, alpha_end, cmap.N)
    alpha = np.expand_dims(alpha, -1)
    if background is not None:
        # Convert background color into an RGBA value
        bg_rgb = mpl.colors.to_rgba(background)[:3]
        # Correct for background color and rescale
        rgb = (rgb - (1 - alpha) * bg_rgb)
        rgb = np.divide(rgb, alpha, out=np.zeros_like(rgb), where=(alpha > 0))
        rgb = np.clip(rgb, 0, 1)
    rgba = np.concatenate((rgb, alpha), axis=1)
    # Create a new Colormap object
    new_name = cmap.name + "_lin_alpha"
    cmap_alpha = mpl.colors.ListedColormap(rgba, name=new_name)
    if register:
        mpl.cm.register_cmap(name=new_name, cmap=cmap_alpha)
    return cmap_alpha

# absolute Luftfeuchtigkeit in g/kg
def g_abs(t: pd.Series, rh: pd.Series, location='Mietraching, Germany', mode='accurate'):
    '''args:
    ---
        t: 
            Lufttemperatur in °C
        rh: 
            rel. Luftfeuchte in %
        location, str:
            Ort zur bestimmung des Luftdrucks
        mode, str {'accurate', 'mean'}
            accurate uses the actual local pressare at a certain PoT.
            mean uses the mean preasse over past 5 years
        '''
    if isinstance(location, str):
        try:
            loc = geocoders.Nominatim(user_agent="RamonsRobusteThesis").geocode(location)
            station = Stations().nearby(loc.latitude, loc.longitude).fetch(1)
            timezone = TimezoneFinder().timezone_at(lng=loc.longitude, lat=loc.latitude)
            #print(f"Luftdruck von Wetterstation {station['name'].values[0]} wird verwendet.")
            if all([hasattr(t, 'index'), hasattr(rh, 'index')]) and mode == 'accurate':
                if all([hasattr(t.index, 'date'), hasattr(rh.index, 'date')]):
                    start = min([rh.index.min(), t.index.min()]).tz_convert('utc').to_pydatetime().replace(tzinfo=None)
                    ende = min([rh.index.min(), t.index.min()]).tz_convert('utc').to_pydatetime().replace(tzinfo=None)
                    refdata = Hourly(station, start, ende).fetch().tz_localize('utc').tz_convert(timezone)
                    p0 = refdata['pres'].reindex_like(rh, 'ffill').mul(100)
                else:
                    mode = 'mean'
                    #print('Kann keinen Zeitraum aus Index schließen')
            else:
                mode = 'mean'
                #print('objekt hat keinen Index')
            p_mean = round(Hourly(station, dt.datetime.now() - dt.timedelta(5*395), dt.datetime.now()).fetch()['pres'].mean()*100)
            #print(p_mean)
        except Exception as e:
            print(f'{e}: Standard Luftdruck wird verwendet...')
            p_mean = 101325             # atmosphärischer Luftdruck in Pa
            p0 = 101325             # atmosphärischer Luftdruck in Pa
            mode = 'mean'
    else:
        p_mean = 101325             # atmosphärischer Luftdruck in Pa
        p0 = 101325             # atmosphärischer Luftdruck in Pa
        #print('Keine StationID übergeben. Mittlerer atmosphärischer Luftdruck wird verwendet.')
    
    rh = rh/100

    if mode == 'accurate': p = p0
    elif mode == 'mean': p = p_mean

    if isinstance(t, (int, float)) & isinstance(rh, (int, float)):
        # absolute Luftfeuchtigkeit in g/kg von Temperatur und Luftfeuchte
        # atmosphärischer Luftdruck in Pa:
        # Wasserdampfsättigungsdruck
        def psat(t):
            if t >= 0:
                return 611*math.exp((17.269*t)/(237.3+t))
            if t < 0:
                return 611*math.exp((21.875*t)/(265.5+t))

        rh=rh/100
        return round( 0.622 * (rh*psat(t))/(p-rh*psat(t)) * 1000 ,2)

    elif isinstance(t, pd.Series) and isinstance(rh, pd.Series):
        pass
    elif isinstance(t, pd.Series) and isinstance(rh, (int, float)):
        rh = pd.Series(rh, index=t.index)

    elif isinstance(rh, pd.Series) and isinstance(t, (int, float)):
        rh = pd.Series(t, index=rh.index)

    elif isinstance(t, pd.DataFrame) and isinstance(rh, pd.DataFrame):
        matchingCols = [col for col in t.columns if col in rh.columns]
        if len(matchingCols) > 0:
            t = t[matchingCols]
            rh = rh[matchingCols]
        else:
            raise TypeError('Keine übereinstimmten Spalten gefunden. Spaltennamen von t und rh müssen übereinstimmen.')
    else:
        raise TypeError('Bitte als pd.Series oder pd.Dataframe (mit übereinstimmenden Spalten) übergeben.')

    p1 = 610.5 * np.exp((17.269 * t.where(t >= 0)).div(237.3 + t.where(t >= 0)))
    p2 = 610.5 * np.exp((21.875 * t.where(t < 0)).div(265.5 + t.where(t < 0)))
    psat = p1.fillna(p2)    # Wasserdampfsättigungsdruck

    return (0.622 * (rh*psat)/((rh*psat*(-1)).add(p, axis=0))*1000).round(2)


################################################ operative Temperatur aus SK- und Lufttemperatur ################################################

def calcTOP(Tair, Tsk, E:float=0.94, D:float=0.07):
    '''
    Berechne die operative Raumtemperatur aus der Luft- und der Schwarzkugeltemperatur

    args:
    -----
        Tair:   Raumlufttemperatur  [°C]
        Tsk:    Schwarzkugeltemperatur [°C]
        E:      Emissionsfaktor der Schwarzkugel [-]
        D:      Durchmesser der Schwarzkugel [m]

    returns:
    ----
        Top:    operative Raumtemperatur [°C]
    '''
    if isinstance(Tair, pd.DataFrame) and isinstance(Tsk, pd.DataFrame):
        cols = []
        for col in Tair.columns:
            if col in Tsk.columns:
                cols.append(col)
        dfs = {}
        for col in cols:
            _Tair = Tair[col]
            _Tsk = Tsk[col]
            MRT_sk = (( ( (_Tsk + 273)**4 ) + ( (0.25*10**8) / E ) * ( abs(_Tsk - _Tair) / D )**(1/4) * (_Tsk - _Tair) )**(1/4) - 273)
            Top = (MRT_sk + _Tair) / 2
            dfs[col] = Top
        df = pd.DataFrame(dfs).round(2)
        return df
    elif isinstance(Tair, pd.Series) and isinstance(Tsk, pd.Series):
        ## MRT,sk
        MRT_sk = (( ( (Tsk + 273)**4 ) + ( (0.25*10**8) / E ) * ( abs(Tsk - Tair) / D )**(1/4) * (Tsk - Tair) )**(1/4) - 273)

        ## Top,sk
        return ((MRT_sk + Tair) / 2)


def getRollingTamb(T_amb, a=0.8):
    return T_amb.resample('D').mean().ewm(alpha=a, min_periods=1, ignore_na=True).mean().reindex(T_amb.index, method='ffill')

################################################ Thermischer Komfort nach DIN EN 15251:2012 - NA ################################################
def KelvinstundenNA(Temp, Tamb, plot=False):
    _data = pd.concat({'T_amb': Tamb, 'Temp': Temp}, axis=1)
    if hasattr(_data.index, 'freqstr'):
        if _data.index.freqstr != 'H':
            _data = _data.resample('H').mean()
    dfs = {}
    if _data.columns.nlevels > 1:
        for col, group in _data.groupby(level=[*range(1,_data.columns.nlevels)], axis=1):
            df = group.droplevel(level=[*range(1,_data.columns.nlevels)], axis=1)
            df = (df
            .join(pd.concat([(df.loc[(df['T_amb'] < 16) & (df['Temp'] > 24), 'Temp'] - 24),
            (df.loc[(df['T_amb'] >= 16) & (df['T_amb'] <= 32) & (df['Temp'] > (20 + 0.25 * df['T_amb'])), 'Temp'] - (20 + 0.25 * df['T_amb'])),
            (df.loc[(df['T_amb'] > 32) & (df['Temp'] > 28), 'Temp'] - 28)]).sort_index().rename('ÜTGS').dropna(how='all'))
            .join(pd.concat([(20 - df.loc[(df['T_amb'] < 16) & (df['Temp'] < 20), 'Temp']),
            ((16 + 0.25 * df['T_amb']) - df.loc[(df['T_amb'] >= 16) & (df['T_amb'] <= 32) & (df['Temp'] < (16 + 0.25 * df['T_amb'])), 'Temp']), 
            24 - (df.loc[(df['T_amb'] > 32) & (df['Temp'] < 24), 'Temp'])]).sort_index().rename('UTGS').dropna(how='all')))

            kh = df[['UTGS', 'ÜTGS']]

            if plot:
                fig, ax = plt.subplots()
                df.where(df.UTGS.notna()).plot(x='T_amb', y='Temp', ls='None', marker='.', ax=ax, label='UTGS')
                df.where(df.UTGS.isna() & df.ÜTGS.isna()).plot(x='T_amb', y='Temp', ls='None', marker='.',color='k', ax=ax, label='Comfortable')
                df.where(df.ÜTGS.notna()).plot(x='T_amb', y='Temp', ls='None', marker='.', color='red', ax=ax, label ='ÜTGS')
                return kh
            dfs[col] = kh   
        kh = pd.concat(dfs, axis=1)
        return kh
    else:
        df = (_data
        .join(pd.concat([(_data.loc[(_data['T_amb'] < 16) & (_data['Temp'] > 24), 'Temp'] - 24),
        (_data.loc[(_data['T_amb'] >= 16) & (_data['T_amb'] <= 32) & (_data['Temp'] > (20 + 0.25 * _data['T_amb'])), 'Temp'] - (20 + 0.25 * _data['T_amb'])),
        (_data.loc[(_data['T_amb'] > 32) & (_data['Temp'] > 28), 'Temp'] - 28)]).sort_index().rename('ÜTGS').dropna(how='all'))
        .join(pd.concat([(20 - _data.loc[(_data['T_amb'] < 16) & (_data['Temp'] < 20), 'Temp']),
        ((16 + 0.25 * _data['T_amb']) - _data.loc[(_data['T_amb'] >= 16) & (_data['T_amb'] <= 32) & (_data['Temp'] < (16 + 0.25 * _data['T_amb'])), 'Temp']), 
        24 - (_data.loc[(_data['T_amb'] > 32) & (_data['Temp'] < 24), 'Temp'])]).sort_index().rename('UTGS').dropna(how='all')))

        kh = df[['UTGS', 'ÜTGS']]
        if plot:
            fig, ax = plt.subplots()
            _data.where(kh.UTGS.notna()).plot(x='T_amb', y='Temp', ls='None', marker='.', ax=ax, label='UTGS')
            _data.where(kh.UTGS.isna() & kh.ÜTGS.isna()).plot(x='T_amb', y='Temp', ls='None', marker='.',color='k', ax=ax, label='Comfortable')
            _data.where(kh.ÜTGS.notna()).plot(x='T_amb', y='Temp', ls='None', marker='.', color='red', ax=ax, label='ÜTGS')  
        return kh

def adaptive_comfort_NA(Tamb=None, Top=None, Tair=None, ax=None, kind='scatter', figsize=None, scale='lin', highlight=False, title=None, annotate=True, print_timespan=False, **kwargs):
    """
    Erstelle ein Diagramm zur Evaluation des thermischen Komoforts nach DIN EN 16798-1:2022 - NA

    Keyword arguments:
        Tamb:
                Außenlufttemperatur
        Temp:
                operative Raumtemperatur
        ax, plt.axes (optional)
                instanz zum plotten des graphen
        mode (optional, default='air')
                'air' für Lufttemperatur, 'op' für operative Temperatur"""
    if ax is None: 
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    if 'cmap' in kwargs:
        cmap = kwargs['cmap']
        del kwargs['cmap']
        if isinstance(cmap, str):
            colormap = plt.cm.get_cmap(cmap)
        elif isinstance(cmap, plt.colormaps.Colormap):
            colormap = cmap
    else:
        colormap = transparentcmap(style.Reds)
    ax.set_prop_cycle('color', [colormap(k) for k in np.linspace(0, 1, 6)])
    mode = 'Top'
    plot_kwargs = dict(marker = '.', linestyle='None', alpha=0.75, label='Raumklima')
    plot_kwargs['markersize'] = 2

    if Tamb is not None:
        if isinstance(Tamb, pd.Series):
            pass
        elif Tamb.shape[1] == 1:
            Tamb = Tamb.squeeze()
        elif Tamb.shape[1] > 1:
            raise ValueError('Bitte nur einen Datensatz für TAMB übergeben.')

        if Top is not None:
            if isinstance(Top, pd.Series):
                pass
            elif Top.shape[1] == 1:
                Top = Top.squeeze()
            elif Top.shape[1] > 1:
                raise ValueError('Bitte nur einen Datensatz für Temp übergeben.')
            df = pd.concat({'Tamb': Tamb,'Temp': Top},axis=1)
            mode = 'Top'
        elif Tair is not None:
            if isinstance(Tair, pd.Series):
                pass
            elif Tair.shape[1] == 1:
                Tair = Tair.squeeze()
            elif Tair.shape[1] > 1:
                raise ValueError('Bitte nur einen Datensatz für Temp übergeben.')
            df = pd.concat({'Tamb': Tamb,'Temp': Tair},axis=1)
            mode = 'Tair'
        else:
            raise ValueError('Es muss eine Raumtemperatur (entweder Top oder Tair) übergeben werden.')
            
        if hasattr(df.index, 'freq'):
            if df.index.freq != 'H':
                df = df.resample('H').mean()

        df.dropna(inplace=True)

        kh = KelvinstundenNA(df['Temp'], df['Tamb'])
        pct_upper = (kh.ÜTGS>2).sum() / df.shape[0]
        pct_lower = (kh.UTGS>2).sum() / df.shape[0]
        ÜTGS = kh.ÜTGS.sum().round()
        UTGS = kh.UTGS.sum().round()
        if annotate:
            if ÜTGS > 0:
                text1 = f'ÜTGS: {ÜTGS:.0f}\n($KH>2: {pct_upper*100:.0f}$ \\si{{\\percent}} der Zeit)'
                ax.text(0.03, 0.97, text1.strip(), ha = 'left', va = 'top', transform=ax.transAxes, fontsize=plt.rcParams['xtick.labelsize'], bbox=dict(boxstyle="round4", ec='none',fc="w"))
            if UTGS > 0:
                text2 = f'UTGS: {UTGS:.0f}\n($KH>2: {pct_lower*100:.0f}$ \\si{{\\percent}} der Zeit)'
                ax.text(0.97, 0.05, text2.strip(), ha = 'right', va = 'bottom', transform=ax.transAxes, fontsize=plt.rcParams['xtick.labelsize'], bbox=dict(boxstyle="round4", ec='none',fc="w"))
        empty = False
    else:
        empty = True

    x = np.linspace(-30,40)
    y=[]
    for t in x:
        if t < 16:
            y.append(22)
        if 16 <= t <= 32:
            y.append(18 + 0.25*t)
        if t > 32:
            y.append(26)

    if 'xlim' in kwargs:
        xlim = kwargs['xlim']
        del kwargs['xlim']
    else: 
        xlim = (-10, 40)

    if 'ylim' in kwargs:
        ylim = kwargs['ylim']
        del(kwargs['ylim'])
    else:
        ylim = (16, 30)

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins='auto', steps=[2,4,5]))
    #ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(base=2)) #(nbins='auto', steps=[1,2,4,10]))
    ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(base=2))
    ax.plot(x, y, c='k',ls = 'dashed', label = 'Komforttemperatur')
    p, = ax.plot(x, [t+2 for t in y],ls='dashdot', c='k')
    p, = ax.plot(x, [t-2 for t in y],ls='dashdot', c='k')
    if highlight:
        ax.fill_between(x, [t+2 for t in y], [t-2 for t in y], color="0.8", label = 'Komfortbereich')
        p = (mpl.patches.Patch(color='0.8', linewidth=0), p)

    grid_kws = dict(axis='both', linestyle='dashed')
    if 'grid_kws' in kwargs:
        grid_kws.update(kwargs['grid_kws'])
    ax.grid(grid_kws)

    ax.set_xlabel('Außenlufttemperatur [°C]')
    if mode == 'Top':
        ax.set_ylabel('operative Raumtemperatur [°C]')
    elif mode == 'Tair':
        ax.set_ylabel('Lufttemperatur [°C]')

    if (title is None) and (print_timespan == False):
        pass
    elif (title is True) and (print_timespan == False):
        ax.set_title('Adaptives Komfortmodell nach DIN EN 16798:2022 - NA', loc='left', y=1.1)
    elif (title is True) and (print_timespan == True):
        ax.set_title(f'Adaptives Komfortmodell nach DIN EN 16798:2022 - NA\n{df.index.min():%d.%m.%y} bis {df.index.max():%d.%m.%y}', loc='left', y=1.1)
    elif isinstance(title, str) and (print_timespan == False):
        ax.set_title(f'{title}')
    elif isinstance(title, str) and (print_timespan == True):
        ax.set_title(f'{title}\n{df.index.min():%d.%m.%y} bis {df.index.max():%d.%m.%y}', loc='left')
        
    if not empty:
        if kind.lower() == 'scatter':
            han, = ax.plot(df['Tamb'], df['Temp'], **plot_kwargs)
        elif kind.lower() == 'hist':
            if scale == 'log':
                cbar = dict(norm=mpl.colors.LogNorm(), vmin=None, vmax=None, cbar_kws=dict(ticks=mpl.ticker.LogLocator(), format=mpl.ticker.ScalarFormatter()))
            else:
                cbar = dict(norm=mpl.colors.Normalize(), vmin=None, vmax=None, cbar_kws=dict(ticks=mpl.ticker.AutoLocator(), format=mpl.ticker.ScalarFormatter()))
            hist = sns.histplot(x=df['Tamb'], y=df['Temp'], cbar=True, ax=ax, cmap=colormap, bins=(np.arange(*xlim, 0.5), np.arange(*ylim, 0.5)), label='Raumklima', **cbar)
            cbar = hist.collections[-1].colorbar
            cbar.outline.set_visible(False)
            cbar.ax.set_title('Stunden')
            han = mpl.patches.Patch(color = colormap(0.5), label='Raumklima')
    else:
        han = mpl.patches.Patch(color = colormap(0.5), label='Raumklima')

    _ms = 10 / plot_kwargs['markersize']
    kws = dict(loc='lower right',bbox_to_anchor=(1,1), markerscale = _ms, ncol=1, frameon=False)
    ax.legend([p, han], ['Komfortbereich', 'Raumklima'], **kws)
    
    fig.tight_layout()
    return ax



################################################ Thermischer Komfort nach DIN EN 16789-1 - Anhang B ################################################
def KelvinstundenEN(Temp, Tamb=None, Tamb_g24=None, kat='II', join=False, plot=False, flat=True):
    if isinstance(kat, list):
        KAT = {key: item for key, item in {'I':2,'II':3,'III':4}.items() if key in kat}
    elif isinstance(kat, str):
        KAT = {kat: {'I':2,'II':3,'III':4}[kat]}
    else:
        KAT = {'I':2,'II':3,'III':4}
    if Tamb_g24 is None and Tamb is not None:
        data = pd.concat({'Tamb_g24': getRollingTamb(Tamb), 'Temp': Temp}, axis=1)
    elif Tamb is None and Tamb_g24 is not None:
        data = pd.concat({'Tamb_g24': Tamb_g24, 'Temp': Temp}, axis=1)
    else:
        raise ValueError('Bitte entweder T_amb oder T_amb_g24 übergeben.')
    if hasattr(data.index, 'freqstr'):
        if data.index.freqstr != 'H':
            data = data.resample('H').mean()
    if isinstance(data.Tamb_g24, pd.DataFrame):
        data = data.where(((data.Tamb_g24 > 10) & (data.Tamb_g24 < 30)).any(axis=1))
    elif isinstance(data.Tamb_g24, pd.Series):
        data = data.where(data.Tamb_g24.between(10, 30))
    else:
        print(type(data), 'impossible to filter...')
    dfs = {}
    if data.columns.nlevels > 1:
        for col, group in data.groupby(level=[*range(1,data.columns.nlevels)], axis=1):
            df = group.droplevel(level=[*range(1,data.columns.nlevels)], axis=1)
            kh = pd.DataFrame()
            for kat, offset in KAT.items():
                kh[f'UTGS_{kat}'] = ((df.Tamb_g24/3 + 18.8 - offset - 1) - df.Temp).where(lambda x: x > 0)
                kh[f'ÜTGS_{kat}'] = (df.Temp - (df.Tamb_g24/3 + 18.8 + offset)).where(lambda x: x > 0)
                if plot:
                    fig, ax = plt.subplots()
                    df.where(kh[f'UTGS_{kat}'].notna()).plot(x='Tamb_g24', y='Temp', ls='None', marker='.', ax=ax, label='UTGS')
                    df.where(kh[f'UTGS_{kat}'].isna() & kh[f'ÜTGS_{kat}'].isna()).plot(x='Tamb_g24', y='Temp', ls='None', marker='.',color='k', ax=ax, label='Comfortable')
                    df.where(kh[f'ÜTGS_{kat}'].notna()).plot(x='Tamb_g24', y='Temp', ls='None', marker='.', color='red', ax=ax, label='ÜTGS')
            if join:
                kh = pd.concat([df, kh], axis=1)
            else:
                kh.columns = kh.columns.str.split('_', expand=True)
                kh = kh.sort_index(axis=1)
            dfs[col] = kh
            if plot: 
                return kh
        kh = pd.concat(dfs, axis=1)
        if isinstance(kat, str):
            kh = kh.droplevel(-1, axis=1)
        return kh
    else: 
        kh = pd.DataFrame()
        for kat, offset in KAT.items():
            kh[f'UTGS_{kat}'] = ((data.Tamb_g24/3 + 18.8 - offset - 1) - data.Temp).where(lambda x: x > 0)
            kh[f'ÜTGS_{kat}'] = (data.Temp - (data.Tamb_g24/3 + 18.8 + offset)).where(lambda x: x > 0)
            if plot:
                fig, ax = plt.subplots()
                data.where(kh[f'UTGS_{kat}'].notna()).plot(x='Tamb_g24', y='Temp', ls='None', marker='.', ax=ax, label=f'UTGS KAT {kat}')
                data.where(kh[f'UTGS_{kat}'].isna() & kh[f'ÜTGS_{kat}'].isna()).plot(x='Tamb_g24', y='Temp', ls='None', marker='.',color='k', ax=ax, label='Comfortable')
                data.where(kh[f'ÜTGS_{kat}'].notna()).plot(x='Tamb_g24', y='Temp', ls='None', marker='.', color='red', ax=ax, label=f'ÜTGS KAT {kat}')
        if join:
            kh = pd.concat([data, kh], axis=1)
        else:
            kh.columns = kh.columns.str.split('_', expand=True)
            kh = kh.sort_index(axis=1)
            if flat:
                kh = kh.droplevel(1, axis=1)
        return kh

def adaptive_comfort_EN(Tamb:pd.Series=None, Top:pd.Series=None, Tair = None, ax:plt.Axes=None, kind='scatter', figsize=None,  highlight=False, scale='log', annotate=True, kat:str = 'II', title:str=None, print_timespan=False, ms:float = None, legend_ms:float = None, **kwargs) -> plt.Axes:
    """
    Erstelle ein Diagramm zur Evaluation des thermischen Komoforts nach DIN EN 16798-1 - Anhang B2.2.

    Achtung! Die Darstellung beinhaltet Temperaturstunden und keine TemperaturGRADstunden.

    Args:
    ----------
        TAMBG24 <pd.Series/pd.DataFrame>:
            gleitender Mittelwert der Außentemperatur über 24h in stündlichen Schritten.
        Temp <pd.Series/pd.DataFrame>:     
            Raumtemperatur. stündliche Mittelwerte.
        ax <plt.axes>:                     
            plt.axes instanz zum plotten des Graphen.
        mode <str>:                         
            'air' für Lufttemperatur, 'op' für operative Temperatur.
        ms <float>:                         
            Markersize in [pt] für die Marker im Plot.
        legend_ms <float>:                  
            Markerscale-factor für die Darstellung der Marker in der legende.
        kat <str>:                          
            Kategorien des Innenkomforts nach DIN EN 16798-1 {'I','II','III'}

    Returns:
    ----------
        ax  <plt.axes>
    """
    if ax is None:  
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    if 'cmap' in kwargs:
        cmap = kwargs['cmap']
        del kwargs['cmap']
        if isinstance(cmap, str):
            colormap = plt.get_cmap(cmap)
        elif isinstance(cmap, plt.colormaps.Colormap):
            colormap = cmap
    else:
        colormap = transparentcmap(style.Reds)
    ax.set_prop_cycle('color', [colormap(k) for k in np.linspace(0, 1, 6)])

    plot_kwargs = {}
    plot_kwargs['markersize'] = 2
    mode = 'Top'
    if 'xlim' in kwargs:
        xlim = kwargs['xlim']
        del kwargs['xlim']
    else: 
        xlim = (10, 30)
    if 'ylim' in kwargs:
        ylim = kwargs['ylim']
        del(kwargs['ylim'])
    else:
        ylim = (16, 32)
    
    for key, item in kwargs.items():
        plot_kwargs[key] = item

    _KAT = {'I':2,'II':3,'III':4}

    if isinstance(kat, str) and kat in _KAT:    
        KAT = {kat: _KAT[kat]}

    elif isinstance(kat, list):
        KAT = {}
        for i in kat:
            if i in _KAT:   KAT[i] = _KAT[i]
    else:   KAT = _KAT

    if Tamb is not None:
        if isinstance(Tamb, pd.Series):
            pass
        elif Tamb.shape[1] == 1:
            Tamb = Tamb.squeeze()
        elif Tamb.shape[1] > 1:
            raise ValueError('Bitte nur einen Datensatz für Tamb übergeben.')
        if Top is not None:
            if isinstance(Top, pd.Series):
                pass
            elif Top.shape[1] == 1:
                Top = Top.squeeze()
            elif Top.shape[1] > 1:
                raise ValueError('Bitte nur einen Datensatz für Top übergeben.')
            df = pd.concat({'T_amb_g24': getRollingTamb(Tamb), 'Temp': Top},axis=1)
            mode = 'Top'
        elif Tair is not None:
            if isinstance(Tair, pd.Series):
                pass
            elif Tair.shape[1] == 1:
                Tair = Tair.squeeze()
            elif Tair.shape[1] > 1:
                raise ValueError('Bitte nur einen Datensatz für Tair übergeben.')
            df = pd.concat({'T_amb_g24': getRollingTamb(Tamb), 'Temp': Tair},axis=1)
            mode = 'Tair'
        else:
            raise ValueError('Es muss eine Raumtemperatur (entweder Top oder Tair) übergeben werden.')
        if hasattr(df.index, 'freq'):
            if df.index.freq != 'H':    df = df.resample('H').mean()
        empty = False
    else:
        empty = True
    # Definiere Wertebereich des Graphen.
    
    linestyle = ['dashdot','dotted','solid']
    x = np.linspace(*xlim)
    for k, key in enumerate(KAT):

        y1 = [(t/3)+18.8-KAT[key]-1 for t in x]
        y2 = [(t/3)+18.8+KAT[key] for t in x]

        p, = ax.plot(x, y1, c='k',ls = linestyle[k])
        p, = ax.plot(x, y2, c='k',ls = linestyle[k])
        if highlight:
            ax.fill_between(x, y1, y2, color='0.8')
            p = (mpl.patches.Patch(color='0.8', linewidth=0), p)
        if kat is not None:
            ax.annotate(f'KAT {key}',xy=(max(x), max(y1)), xycoords='data', xytext=(-5, -10), textcoords='offset points', ha='right', va='top', fontsize=plt.rcParams['xtick.labelsize'])
            ax.annotate(f'KAT {key}', xy=(max(x), max(y2)), xycoords='data', xytext=(-5, -10), textcoords='offset points', ha='right', va='top', fontsize=plt.rcParams['xtick.labelsize'])
        else: 
            ax.annotate(f'KAT {key}',xy=(max(x), max(y1)), xycoords='data', xytext=(5, 0), textcoords='offset points', ha='left', va='center', fontsize=plt.rcParams['xtick.labelsize'])
            ax.annotate(f'KAT {key}', xy=(max(x), max(y2)), xycoords='data', xytext=(5, 0), textcoords='offset points', ha='left', va='center', fontsize=plt.rcParams['xtick.labelsize'])
    y0 = [(t/3)+18.8 for t in x]
    ax.plot(x, y0, c='k',ls = 'dashed', label = 'Komforttemperatur')
    grid_kws = dict(axis='both', linestyle='dashed')
    if 'grid_kws' in kwargs:
        grid_kws.update(kwargs['grid_kws'])
    ax.grid(grid_kws)
    ax.set_xlabel('gleitender Mittelwert der Außenlufttemperatur [°C]')
    ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins='auto', steps=[2,4,5])) #(nbins='auto', steps=[1,2,4,10]))
    ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(base=2))

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    if mode=='Tair':
        ax.set_ylabel('Raumlufttemperatur [°C]')
    if mode=='Top':
        ax.set_ylabel('operative Raumtemperatur [°C]')
        
    if (title is None) and (print_timespan == False):
        pass
    elif (title is True) and (print_timespan == False):
        ax.set_title('Adaptives Komfortmodell nach DIN EN 16798-1 - Anhang B2.2',loc='left', y=1.1)
    elif (title is True) and (print_timespan == True):
        ax.set_title(f'Adaptives Komfortmodell nach DIN EN 16798-1 - Anhang B2.2\n{df.index.min():%d.%m.%y} bis {df.index.max():%d.%m.%y}',loc='left', y=1)
    elif isinstance(title, str) and (print_timespan == True):
        ax.set_title(f'{title}\n{df.index.min():%d.%m.%y} bis {df.index.max():%d.%m.%y}')
    elif isinstance(title, str) and (print_timespan == False):
        ax.set_title(f'{title}')

    if not empty:
        df=df[(df.T_amb_g24 >10) & (df.T_amb_g24 < 30)].dropna()
        if df.shape[0] == 0:
            ax.text(0.5,0.5, 'Alle Datenpunkte außerhalb des Definitionsbereich:\n' + r'$10°C < T_{amb,g24} < 30°C$',style='normal', ha = 'center', va = 'center', transform=ax.transAxes, bbox=dict(boxstyle="round", fc="w", ec="0.5", alpha=0.9))
        else:
            if kind.lower() == 'scatter':
                han, = ax.plot(df['T_amb_g24'], df['Temp'], marker = '.', linestyle='None', alpha=0.75,label='Raumklima', **plot_kwargs)
            elif kind.lower() == 'hist':
                if scale == 'log':
                    cbar = dict(norm=mpl.colors.LogNorm(), vmin=None, vmax=None, cbar_kws=dict(ticks=mpl.ticker.LogLocator(), format=mpl.ticker.ScalarFormatter()))
                else:
                    cbar = dict(norm=mpl.colors.Normalize(), vmin=None, vmax=None, cbar_kws=dict(ticks=mpl.ticker.AutoLocator(), format=mpl.ticker.ScalarFormatter()))
                hist = sns.histplot(x=df['T_amb_g24'], y=df['Temp'], ax=ax, cmap=colormap, cbar=True, bins=(np.arange(*xlim, 0.5), np.arange(*ylim, 0.5)), **cbar)
                cbar = hist.collections[-1].colorbar
                cbar.outline.set_visible(False)
                cbar.ax.set_title('Stunden')
                han = mpl.patches.Patch(color = colormap(0.5), label='Raumklima')
            if annotate:
                results = KelvinstundenEN(Tamb_g24 = df['T_amb_g24'], Temp=df['Temp'], kat=kat, flat=False)
                pct = results.gt(2).sum().unstack(0).div(df.dropna().shape[0])
                results = results.sum().round().unstack(0)
                text = {'ÜTGS':[], 'UTGS':[]}
                for key, s in results.iteritems():
                    for _kat, item in s.iteritems():
                        if item > 0:
                            text[key].append(f'KAT {_kat}: {item:.0f}\n($KH>2: {pct.at[_kat, key]*100:.0f}$ \\si{{\\percent}} der Zeit)')
                    text[key] = '\n'.join(text[key])
                if len(text['UTGS']) > 0:
                    ax.text(0.97, 0.05, text['UTGS'].strip(), style='normal', ha = 'right', va = 'bottom', transform=ax.transAxes, fontsize=plt.rcParams['xtick.labelsize'],bbox=dict(boxstyle="round4", ec='none',fc="w"))
                if len(text['ÜTGS']) > 0:
                    ax.text(0.03, 0.97, text['ÜTGS'].strip(), style='normal', ha = 'left', va = 'top', transform=ax.transAxes, fontsize=plt.rcParams['xtick.labelsize'],bbox=dict(boxstyle="round4", ec='none',fc="w"))
    else:
        han = mpl.patches.Patch(color = colormap(0.5), label='Raumklima')
    _ms = 10 / plot_kwargs['markersize']
    ax.legend([p, han], ['Komfortbereich', 'Raumklima'], loc='lower right', markerscale=_ms, ncol=1, bbox_to_anchor=(1,1), frameon=False)
    fig.tight_layout()
    return ax

class hxdiagramm():
    def __init__(self, temp=None, rh=None, ax=None, figsize=None, kind='hist', cbar=True, title=True, **kwargs):
        self.figsize = figsize
        self.ndatasets = 0
        self.g_mode = 'mean'
        self._plotkwargs = {}
        self._plotkwargs['markersize'] = 1
        self._plotkwargs['alpha'] = 1
        self._datahandles = []
        self._datalabels = []
        self._kind = kind
        self._rc = plt.rcParams
        self._comfdesc = 'annot'
        self.cbar = cbar
        self.scale = 'log'
        if self.scale == 'log':
            self.norm = mpl.colors.LogNorm()
        else:
            self.norm = mpl.colors.Normalize()
        self._cmap = []
        if ax is None:
            self.fig, self.ax = plt.subplots(figsize=self.figsize)
        else:
            self.ax = ax
            self.fig = plt.gcf()
        _fig, _ax = self.fig, self.ax

        if 'print_timespan' in kwargs:
            self.print_timespan = kwargs['print_timespan']
            self.timespan = []
            del kwargs['print_timespan']
        else:
            self.print_timespan = False

        if 'cmap' in kwargs:
            cmap = kwargs['cmap']
            del kwargs['cmap']
        else:
            cmap = style.Blues

        if isinstance(cmap, str):
            self.colormap = plt.get_cmap(cmap)
            _ax.set_prop_cycle('color', [self.colormap(k) for k in np.linspace(0, 1, 6)])
        elif isinstance(cmap, mpl.colors.Colormap):
            self.colormap = cmap
            _ax.set_prop_cycle('color', [self.colormap(k) for k in np.linspace(0, 1, 6)])
        elif isinstance(cmap, list):
            _ax.set_prop_cycle('color', cmap)
            self.colors = cmap

        if 'xlim' in kwargs:
            self._xlim = kwargs['xlim']
            del kwargs['xlim']
        else:
            self._xlim = (15, 30)

        if 'ylim' in kwargs:
            self._ylim = kwargs['ylim']
            del kwargs['ylim']
        else:
            self._ylim = (0, 20)

        #Ränder der Grafik
        min_x, max_x = self._xlim
        min_y, max_y = self._ylim

        _n_ = 150

        _df = pd.concat({_rh_: g_abs(pd.Series(np.linspace(min_x,max_x, _n_), index=np.linspace(min_x,max_x, _n_)), _rh_, mode='mean') for _rh_ in np.linspace(10,100,10)}, axis=1)

        for _rH_, col in _df.iteritems():
            _ax.plot(col, 'k-' , linewidth = self._rc['lines.linewidth'] * 0.5, zorder=1)
            if col.loc[max_x] <= max_y:
                _ax.annotate(text= f'${_rH_:.0f}\\%$', xy = (max_x,col.loc[max_x]), xytext = (3, 0),fontsize=self._rc['xtick.labelsize'], textcoords='offset pixels', ha = 'left', va = 'center')
            if col.loc[max_x] > max_y:
                x = np.interp(max_y, col, col.index)
                offset = 3
                _ax.annotate(xy = (x, max_y), text=f'${_rH_:.0f}\\%$', ha = 'left', va = 'bottom', fontsize=self._rc['xtick.labelsize'], rotation = 45, xytext = (offset, 0), textcoords='offset pixels')

        maxaf = 11.5

        comdf = pd.Series(np.linspace(20,26,21), name='temp').to_frame()
        comdf['rH_min'] = 30
        comdf['rH_max'] = 65
        comdf['g_min'] = g_abs(comdf.temp, comdf.rH_min, mode='mean')
        comdf['g_max'] = g_abs(comdf.temp, comdf.rH_max, mode='mean').where(lambda x: x<maxaf, maxaf)
        _ax.fill_between(x = comdf.temp, y1 = comdf.g_min, y2 = comdf.g_max, fc ='k', ec='None', alpha=0.25)
        self.comfort = comdf
        
        _ax.set(ylim=self._ylim, xlim=self._xlim, xlabel=r'Lufttemperatur [\si{\celsius}]', ylabel=r'Absolute Luftfeuchte [\si{\gram\per\kilogram}]')

        if title is None:
            self.title = ''
        elif title is True:
            self.title = 'H,x - Diagramm nach DIN 1946-6'
        elif isinstance(title, str):
            self.title = title

        # if title is None:
        #     pass
        # elif title is True:
        #     _ax.set_title('H,x - Diagramm nach DIN 1946-6', loc = 'left')
        # elif isinstance(title, str):
        #     _ax.set_title(title, loc = 'left')

        _ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(steps=[2,4,5,10]))
        _ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(steps=[2,4,5,10]))
        _ax.xaxis.set_major_formatter('{x:2n}')
        _ax.yaxis.set_major_formatter('{x:2n}')
        _ax.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(1))
        _ax.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(1))
        grid_kws = dict(axis='both', linestyle='dashed')
        if 'grid_kws' in kwargs:
            grid_kws.update(kwargs['grid_kws'])
        _ax.grid(**grid_kws)
        ax1 = _ax.twinx()
        ax1.sharey(_ax)
        ax1.yaxis.set_visible(False)
        ax1.set_zorder(100)

        if self._comfdesc == 'annot':
            _ax.annotate('Behaglichkeitsbereich', ha='center', va='center', xy=(comdf.temp.median(), comdf[comdf.temp == comdf.temp.median()].g_min.min()), xytext=(0.75,0.1), xycoords='data', size=self._rc['xtick.labelsize'], textcoords=_ax.transAxes, bbox=dict(boxstyle="round4", ec='none',fc="w"), arrowprops=dict(arrowstyle="-|>", connectionstyle="arc3,rad=-0.2", fc="w"))
        elif self._comfdesc == 'leg':
            ax1.legend([mpl.patches.Patch(color='k', linestyle='dashed', alpha=0.25, lw=2)], ['Behaglichkeitsbereich'], loc='lower right', frameon=True)

        if (temp is not None) and (rh is not None):
            self._addData(temp=temp, rh=rh)

        x0, y0 = self.comfort.at[0, 'temp'], self.comfort.at[0, 'g_min']
        x1, y1 = self.comfort.at[0, 'temp'], self.comfort.at[0, 'g_max']
        x2, y2 = self.comfort[self.comfort.g_max == self.comfort.g_max.max()].temp.min(), self.comfort.g_max.max()
        x3, y3 = self.comfort[self.comfort.g_max == self.comfort.g_max.max()].temp.max(), self.comfort.g_max.max()
        x4, y4 = self.comfort.temp.max(), self.comfort[self.comfort.temp == self.comfort.temp.max()].g_min.max()
        x5, y5 = self.comfort.temp.max(), self.comfort[self.comfort.temp == self.comfort.temp.max()].g_min.max()

        xs = [x0, x1, x2, x3, x4, x5, x0]
        ys = [y0, y1, y2, y3, y4, y5, y0]
        
        ax1.plot(xs, ys, ls='dashed', color='k', zorder=100)
        if self.print_timespan == True:
            self.title = f'{self.title}\n{min(self.timespan):%d.%m.%y} bis {max(self.timespan):%d.%m.%y}'
        _ax.set_title(self.title, loc = 'left')
        _fig.tight_layout()
        if kind == 'hist' and cbar:
            self._add_colorbar()

    def _addData(self, temp, rh, label=None):
        if isinstance(temp, pd.DataFrame):
            if temp.shape[1] == 1:
                temp = temp.squeeze()
        if isinstance(rh, pd.DataFrame):
            if rh.shape[1] == 1:
                rh = rh.squeeze()
        if isinstance(temp, (pd.DataFrame)) and isinstance(rh, (pd.DataFrame)):
            self.datasets = [col for col in temp if col in rh.columns]
            self.ndatasets += len(self.datasets)
            self._plotkwargs['alpha'] = 1/self.ndatasets
        elif isinstance(temp, pd.Series) and isinstance(rh, pd.Series):
            self.ndatasets = 1
        
        self.data = pd.concat({'temp': temp, 'rH': rh, 'g_abs': g_abs(temp, rh, mode=self.g_mode)}, axis=1)
        if self.ndatasets > 1:
            self.data = self.data.reorder_levels([*range(1, self.data.columns.nlevels), 0], axis=1).sort_index(axis=1)
        
        if hasattr(self.data.index, 'freqstr'):
            if self.data.index.freqstr != 'H':
                print(f'Timestep is {self.data.index.freq}. Resampling to Hour...')
                self.data = self.data.resample('H').mean().round(2)

        self._iscomforable()
        if self.ndatasets > 1:    
            for label, df in self.data.groupby(level=[*range(self.data.columns.nlevels-1)], axis=1):
                self._plotData(df=df[label].dropna().astype(float, errors='raise'), label = label)
        else:
            self._plotData(df=self.data.dropna().astype(float, errors='raise'), label='Raumklima')
        return self.ax

    def _iscomforable(self):
        if self.ndatasets > 1:
            glvl = [*range(self.data.columns.nlevels-1)]
            for colid, df in self.data.groupby(level=glvl, axis=1):
                df = df[colid]
                if isinstance(colid, tuple):
                    new_colid = colid+('comfort',)
                elif isinstance(colid, str):
                    new_colid = (colid, 'comfort')
                else:
                    print(type(colid))    
                self.data.loc[:,new_colid] = df.temp.between(self.comfort.temp.min(), self.comfort.temp.max()) & df.rH.between(self.comfort.rH_min.min(), self.comfort.rH_max.max()) & df.g_abs.lt(self.comfort.g_max.max())
        else:
            self.data.loc[:,'comfort'] = self.data.temp.between(self.comfort.temp.min(), self.comfort.temp.max()) & self.data.rH.between(self.comfort.rH_min.min(), self.comfort.rH_max.max()) & self.data.g_abs.lt(self.comfort.g_max.max())
        self.data.sort_index(axis=1, inplace=True)

    def _add_colorbar(self):
        plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9)
        fig = self.fig
        bbox = self.ax.get_position() 
        # [x0 (left), y0 (bottom), x1 (right), y1 (top)] of the axis.
        width = 0.02
        eps = 0.075 #margin between plot and colorbar
        # [left most position, bottom position, width, height] of color bar.
        #norm = mpl.colors.Normalize(vmin=self.vmin, vmax=self.vmax)
        box = [bbox.x1 + eps, bbox.y0, width, bbox.height]
        printtitle=True
        for cmap in self._cmap:
            cax = fig.add_axes(box)
            if printtitle:
                cax.set_title('Stunden')
                printtitle=False
            cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=self.norm, cmap=cmap),cax=cax)
            cbar.ax.set_yticks([])
            cbar.outline.set_visible(False)
            bbox = cax.get_position()
            box = [bbox.x1, bbox.y0, width, bbox.height]
        if self.scale == 'log':
            cbar.ax.yaxis.set_major_locator(mpl.ticker.LogLocator())
        else:
            cbar.ax.yaxis.set_major_locator(mpl.ticker.AutoLocator())
        cbar.ax.yaxis.set_major_formatter(mpl.ticker.ScalarFormatter())
        #cbar.ax.set_ylabel('Stunden', rotation=0, y=0, x=0, ha='right', va='center')
        return self.ax

    def _transparentcmap(self, cmap):
        my_cmap = cmap(np.arange(cmap.N))
        my_cmap[:,-1] = np.linspace(0, 1, cmap.N)
        my_cmap = mpl.colors.ListedColormap(my_cmap)
        return my_cmap

    def _plotData(self, df, label):
        self.timespan.append(df.index.min())
        self.timespan.append(df.index.max())
        n_comf = df['comfort'].mean()
        label = f'{label}: ${n_comf*100:.0f}\\%$'
        if self._kind == 'scatter':
            h, = self.ax.plot(df['temp'], df['g_abs'], marker = '.', linestyle = 'None', **self._plotkwargs)
            self._datahandles.append(h)
            self._datalabels.append(label)
        elif self._kind == 'hist':
            if hasattr(self, 'colormap'):
                _cmap = self._transparentcmap(self.colormap)
                self._cmap.append(_cmap)
            else:
                _cmap = self._transparentcmap(sns.light_palette(self.colors.pop(), as_cmap=True, reverse=False))
                self._cmap.append(_cmap)
            sns.histplot(x=df['temp'], y=df['g_abs'],vmin=None, vmax=None, norm=self.norm, cmap=_cmap, bins=(np.arange(*self._xlim, 0.5), np.arange(*self._ylim, 0.5)))
            self._datahandles.append(mpl.patches.Patch(color=_cmap(255)))
            self._datalabels.append(label)
        self._plotDataLegend()
        self.fig.tight_layout()

    def _plotDataLegend(self, mode='ax', **legend_kws):
        if hasattr(self, 'leg'):
            self.leg.remove()
        obj = getattr(self, mode)
        leg_kws={'loc':'upper left', 'frameon':False, 'title':r'\si{\percent} der Zeit behaglich', 'frameon':True, 'edgecolor':'w', 'facecolor':'w'}
        if mode == 'fig':
            leg_kws['bbox_to_anchor'] = (0.1,1)
        if len(self._datahandles) > 0:
            for _han in self._datahandles:
                if isinstance(_han, plt.Line2D):
                    _han.set_alpha(1)
                    leg_kws['markerscale'] = 10 / _han.get_markersize()
                else:
                    leg_kws['markerscale'] = 1
            leg_kws.update(legend_kws)
            self.leg = obj.legend(self._datahandles, self._datalabels, **leg_kws)
        else:
            leg_kws.update(legend_kws)
            self.leg = obj.legend(**leg_kws)
        self.leg._legend_box.align = "left"

    def savefig(self, *args, **kwargs):
        self.fig.savefig(*args, **kwargs)
    
    def set_title(self, *args, title=True, fig=True, **kwargs):
        if title==False:
            self.ax.set_title(None)
            self._plotDataLegend(mode='fig')
        else:
            if fig:
                self.fig.suptitle(*args, x=0.08, ha='left', **kwargs)
            else:
                self.fig.suptitle(self.ax.get_title(loc='left'), x=0.08, ha='left')
                self.ax.set_title(*args, **kwargs)
        return self.ax