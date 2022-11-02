
import locale
locale.setlocale(locale.LC_ALL, 'de_DE.UTF-8')
locale.setlocale(locale.LC_NUMERIC, 'de_DE.UTF-8')

import datetime as dt
import string
import os
import time
import logging

import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

import numpy as np
import pandas as pd
from pandas import IndexSlice as IDX

from . import style
from .style import BUID, APPS, ROOMS
from . import toolbox as tb
from .toolbox.utils import aggMINMAX, setup_logger

from sklearn.cluster import KMeans
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from pythermalcomfort.models import pmv_ppd
from pythermalcomfort.utilities import v_relative, clo_dynamic

def getOccupied(df):
    data = df.copy()
    if hasattr(data.index, 'freq'):
        _freq_ = data.index.freqstr
        _anw = getData(value='Anwesenheit').resample(_freq_).quantile(.5, interpolation='nearest').droplevel([2,3], axis=1)
        for col, group in data.groupby(level=['bui', 'app'], axis=1):
            data[col] = data.loc[:,col].where(_anw[col]>0)
    return data
    
def getHeiztage():
    data = removeUnoccupied(getData(value='HK_IO', app=['N', 'S'])).groupby(level=[0,1,2],axis=1).median().resample('H').mean().resample('D').mean()

    df = data.rolling('3D', center=True).mean().round(2)
    dfs = []
    for (bui, app, room), group in df.groupby(level=[0,1,2], axis=1):
        mydata = group[bui, app, room].dropna()
        cluster_id = KMeans(2).fit_predict(mydata.values.reshape(-1,1))
        cluster_means = pd.Series({ii: mydata[cluster_id==ii].mean() for ii in np.unique(cluster_id)})
        status = {cluster_means.idxmax(): True, cluster_means.idxmin(): False}
        dfs.append(pd.Series(cluster_id, index=mydata.index).replace(status).rename((bui, app, room, 'HK_IO')))
    return removeUnoccupied(pd.concat(dfs, axis=1)).reindex(getData(value='HK_IO', app=['N', 'S']).index, method='ffill').where(getData(value='HK_IO', app=['N', 'S']).notna()).rename_axis(getData().columns.names, axis=1).sort_index().sort_index(axis=1).dropna(how='all').dropna(how='all', axis=1).copy()

def removeHolidays(mode='DB', plot=False, **kwargs):   
    """
    mode:
        -   'DB': returns filtered Database
        -   'filter':   returns boolean array where True are days classified as not away times.
    plot:
        -   plotting some graphs
    kwargs:
        -   takes bui, app, room keywords to filter DB
    """ 
    filter_kws = {'app':['N','S']}
    filter_kws.update(kwargs)
    data = removeUnoccupied(getData(value='Anwesenheit', app=['N', 'S'])).groupby(level=[0,1],axis=1).median().resample('H').mean().resample('D').mean()
    df = data.rolling('3D', center=True).mean().round(2)
    mydata = df.stack([0,1]).dropna()
    cluster_id = KMeans(2, n_init=100).fit_predict(mydata.values.reshape(-1,1))
    cluster_means = pd.Series({ii: mydata[cluster_id==ii].mean() for ii in np.unique(cluster_id)})
    status = {cluster_means.idxmax(): False, cluster_means.idxmin(): True}
    tags = {True: {'label':'Abwesend', 'color':style.clrs[1]}, False: {'label':'Anwesend', 'color':style.clrs[0]}}
    sig = pd.Series(cluster_id, index=mydata.index).map(status).rename('Anwesenheit').unstack(['bui', 'app']).sort_index()
    if plot:
        bins = np.linspace(mydata.min(), mydata.max(), 24)
        fig, ax = plt.subplots(**style.size(aspect=0.4))
        for ii, stat in status.items():
            ax.hist(mydata[cluster_id==ii], bins=bins, **tags[stat])
        ax.set(ylabel=r'abs. Häufigkeit', xlabel=r'Anteil Heizungsaktivierung pro Tag')
        ax.legend()
        fig.tight_layout()
        df = pd.concat([data, sig], keys=['Anwesenheit', 'Abwesend'], axis=1).reorder_levels([1,2,0], axis=1).sort_index(axis=1)
        fig, axs = plt.subplots(6, 1, **style.size(aspect=1.5), sharey=True, sharex=True)
        for b, ((bui, app), group) in enumerate(df.groupby(level=[0,1], axis=1)):
            axs[b].set_title(f'{style.NAMES[bui]} | Wohnung {style.NAMES[app]}')
            axs[b].set_ylabel(r'Anteil Anwesenheit')
            axs[b].fill_between(x=group[bui][app].index, y1=group[bui][app].Anwesenheit, alpha=1, label='mittlere Anwesenheit')
            axs[b].fill_between(x=group[bui][app].index, y1=0, y2=1, where=group[bui][app].Abwesend&group.isna().mean(axis=1).eq(0), color=style.red, alpha=0.5, label='Abwesend')
            #axs[b].fill_between(x=group[bui][app].index, y1=0, y2=1, where=group.isna().mean(axis=1), color='k', alpha=0.25, label='NA')
        style.uniqueLegend(axs, fig, ncol=3)
        fig.tight_layout()
    sig = removeUnoccupied(sig.reindex(DB.index, method='ffill')).dropna(how='all', axis=1).fillna(False).sort_index(axis=1)
    if mode == 'DB':
        return pd.concat([group.where(sig[bui][app]==0) for (bui, app), group in getData(**filter_kws).groupby(['bui', 'app'], axis=1)], axis=1)
    elif mode == 'filter':
        return sig

def getData(*args, start=None, end=None, occupied=False, **kwargs):
    '''
    Entweder ein Tuple mit (level, arguments) oder ein dict {levelname: arguments} übergeben
    '''
    global DB
    if 'DB' not in globals():
        getDB()
    m = []

    for (key, item) in args:
        if isinstance(key, int) and key in range(DB.columns.nlevels):
            if isinstance(item, str):
                m.append(DB.columns.get_level_values(key) == item)
            else: 
                m.append(DB.columns.get_level_values(key).isin(item))

    for key, item in kwargs.items():
        if key in DB.columns.names or int(key) in DB.columns.names:
            if isinstance(item, str):
                m.append(DB.columns.get_level_values(key) == item)
            else: 
                m.append(DB.columns.get_level_values(key).isin(item))
    if len(m) == 0:
        if occupied:
            return getOccupied(DB)
        else:
            return DB
    else:
        if occupied:
            return getOccupied(DB.loc[start:end,DB.columns[np.vstack(m).T.all(axis=1)]])
        else:
            return DB.loc[start:end,DB.columns[np.vstack(m).T.all(axis=1)]]
    
def getHoursOccupied(**kwargs):
    return getData(value='Anwesenheit', **kwargs).resample('H').median().sum().reset_index([2,3], drop=True)

def estimateOperativeTemperature(freq='h', mode='fillgaps', model='linear', split_bui=True, plot=False, export=False, **kwargs):
    """
    freq:
        Pandas Freq-String für das Resapling
    
    mode:
        -   'fillgaps':   Behält die Messdaten und füllt nur die Lücken mit der geschätzen Temperatur auf.
        -   'NA':         schätzt die operative Temperatur nur, wo sie nicht gemessen wurde und gibt einen DataFrame mit NA anstelle eines existierenden Messwertes aus.   
        -   'pred':       Schätz die operative Temperatur für den gesamten Zeitraum.
    
    model:
        -   'linear':     
        Einfaches Modell ohne Kreuzvalidation; Verwenet sklearn.linear_model.LinearRegression() zur Schätzung
        -   'CV':         
        Wendet eine Kreuzvalidation an; Verwendet sklearn.linear_model.RidgeCV() zur Schätzung
    
    plot:   boolean
        Plottet die Regressionsgraphen.
    
    '
    """
    if 'DB' not in globals():
        getDB()
    # Load the dataset
    fil_kws = {'room': ['SWK'], 'value': ['Tair', 'Top']}
    _top = []
    db = getData(**fil_kws).resample('min').mean().stack([0,1,2]).dropna()
    if not split_bui:
        X = db['Top'].to_numpy().reshape(-1, 1)
        y = db['Tair'].to_numpy().reshape(-1, 1)
    if plot:
        fig, axs = plt.subplots(3,1, sharex=True, sharey=True, **style.size(subplots=(2,1)))
        
    for b, (bui, data) in enumerate(db.groupby(level=['bui'], axis=0)):
        if split_bui:
            X = data['Top'].to_numpy().reshape(-1, 1)
            y = data['Tair'].to_numpy().reshape(-1, 1)

        if model == 'linear':
            regr = linear_model.LinearRegression().fit(X, y)
        elif model == 'CV':
            regr = linear_model.RidgeCV().fit(X, y)
        else:
            raise ValueError('model must be linear or CV.')
        if plot:
            # Make predictions using the testing set
            testdata = getData(bui=bui, room = ['SZ', 'WZ'], value = ['Tair', 'Top']).resample(freq).mean().stack([0,1,2]).dropna().round(1)
            X_test = testdata['Top'].to_numpy().reshape(-1, 1)
            y_test = testdata['Tair'].to_numpy().reshape(-1, 1)
            y_pred = regr.predict(X_test).round(1)
            ax = axs[b]
            ax.set_title(f'{BUID[bui]}')
            # The coefficients
            print("Ermittleter Faktor: \n", regr.coef_)
            # The mean squared error
            print("Mittleres Residuenquadrat: %.2f" % mean_squared_error(y_test, y_pred))
            # The coefficient of determination: 1 is perfect prediction
            print("$R^2$: %.2f" % r2_score(y_test, y_pred))
            
            ax.scatter(X_test, y_test, s=1, color='k', marker='+', label=f'{bui}: beobachtete Werte')
            ax.plot(X_test, y_pred, color="blue", linewidth=1, label=f'{bui}: Regressionsgerade')
            text = '\n'.join([
                f"Mittleres Residuenquadrat: {mean_squared_error(y_test, y_pred).round(3):3n} $\\si{{\\kelvin}}$", 
                f"Bestimmtheitsmaß $R^2$: {r2_score(y_test, y_pred).round(3):3n}",
                f"Ermittelter Faktor: {regr.coef_[0][0].round(2):3n}"])
            ax.set(ylabel=r'Lufttemperatur $[\si{\celsius}]$', xlabel=r'operative Raumtemperatur $[\si{\celsius}]$')
            ax.text(0.03,1, text, transform=ax.transAxes, ha='left', va='top')
            ax.legend(ncol=2, frameon=False, bbox_to_anchor = (1, .98), loc='lower right')
            ax.grid()
            fig.tight_layout()
            if export:
                style.toTex('pp','linreg_top', fig)

        _Tair = getData(bui = bui, value='Tair').resample(freq).mean().droplevel(3, axis=1)
        if mode == 'NA':
            _Top = getData(bui=bui, value='Tair').resample(freq).mean().droplevel(3, axis=1).reindex_like(_Tair)
            _Tair = _Tair.where(_Top.isna())

        Top = _Tair * regr.coef_[0][0]
        
        if mode == 'fillgaps':
            _Top = getData(bui=bui, value='Tair').resample(freq).mean().droplevel(3, axis=1).reindex_like(_Tair)
            Top = _Top.fillna(Top)

        if mode == 'pred':
            pass

        _top.append(Top)
    Top = pd.concat({'Top': pd.concat(_top, axis=1).round(1)}, axis=1).reorder_levels([1,2,3,0], axis=1).rename_axis(['bui', 'app', 'room', 'value'], axis=1)
    return Top

def getLowerUpperQuantile(df, _id, qt=0.05):
    '''
    Entferne Datensätze die Außerhalb des [qt, 1-qt]-Intervalls der GESAMTEN Messdaten sind.

    Hintergrund:
    ---
    Die trh-Sensoren vertauschen gelegentlich den rH mit dem Tair Messwert. Da es nur sporadisch Auftritt werden diese Fehler einfach aussortiert.
    '''
    sensors = {'Tair': 'trh_Tair', 'RH':'trh_RH', 'Thk':'pt_Thk'}
    return (df.filter(like=sensors[_id]).min().quantile(qt, interpolation = 'nearest'), df.filter(like=sensors[_id]).max().quantile(1-qt, interpolation = 'nearest'))

def getTset(_df, qt=0.75):
    '''
    Definiere die Einstellung des Heizkörpers als qt (default: 0.75)-Quantil der Raumlufttemperatur in einem gegebenen Zeitraum.
    ---
    Hintergrund:
    Der Heizkörper heizt sich solange auf, bis die eingestellte Raumtemperatur erreicht ist, dann fährt er wieder runter um die Temperatur einzupendeln.
    das gewählte Quantil beschreibt die Temperatur die nur von 1-qt überschritten wird. Das gewährt eine gewisse Toleranz.
    '''
    return _df.quantile(qt, interpolation='nearest').round()

def scan_eb_folder(dir_db = './_db/eb-database'):
    global files
    files = {}
    for meter in ['tf','em']:
        if meter not in files:
            files[meter] = {} 
        for bui in ['LB','MW','MH','WD','PM']:
            path = os.path.join(dir_db,bui)
            if not os.path.isdir(path):
                os.makedirs(path)
            for fn in os.listdir(path):
                names = fn.split('_')
                if names[1] == meter:
                    if bui not in files[meter]:
                        files[meter][bui] = {}
                    files[meter][bui][fn.rsplit('_',1)[-1].split('.')[0]] = os.path.join(os.path.join(path), fn)
    return files

#############################       LESE DATENSÄTZE EIN         ###################################
def getFensterDatenbank(bui=None, app=None, IO=True):
    if 'IND' not in globals():
        getIND()
    _Fenster = {}
    if bui is not None:
        if app is None:
            db = IND[bui]
        else:
            db = IND[bui][app]
    else:
        db = IND
    _nlvl = db.columns.nlevels
    for cols, group in db.filter(like = 'Fenster').groupby(level=[*range(_nlvl)], axis=1):
        group = group.droplevel([*range(_nlvl-1)], axis=1).squeeze()
        # Lege für jeden Raum eine Liste an...
        if len(group.dropna()) > 0:
            s = group.rolling('3min').quantile(0.5, interpolation='nearest').where(lambda s: (s.resample('D').mean().asfreq('min', 'ffill') < 1)).where(group.notna())
            if IO:
                df = s.fillna(-1).reset_index()
                df = df.groupby((df[cols[-1]] != df[cols[-1]].shift()).cumsum()).agg({'Datetime': ['min', 'max', aggMINMAX], cols[-1]:'mean'}).set_axis(['Start', 'End', 'Dauer', 'Zustand'], axis=1).where(lambda df: (df.Zustand>0) & (df.Dauer.between(pd.Timedelta(minutes=1), pd.Timedelta(days=1)))).dropna(how='all')
                _Fenster.update({(cols):pd.concat([s.rename('Fenster'), (df.set_index('Start')['Zustand'] > 0).rename('_AUF'), (df.set_index('End')['Zustand'] > 0).rename('_ZU')], axis=1)})
            else:
                _Fenster.update({(*cols, 'Fenster'): s})
    if IO:
        return pd.concat(_Fenster, axis=1)
    else:
        return pd.concat(_Fenster, axis=1).groupby(level=[-3,-1], axis=1).sum(min_count=1)

def updateKorrektur_lin(plot=True, toTex=False):
    if 'IND' not in globals():
        getIND()
    coefs = {}
    for sen in ['Tair', 'RH']:
        data = IND.loc[:,pd.IndexSlice[:,'O', 'SWK',:]].filter(like=sen).filter(like='trh').rename_axis(['bui', 'app', 'room', 'value'], axis=1).stack([0,1,2]).reset_index().rename(columns={'m_trh1_Tair (°C)': 'trh_ref', 'm_trh_Tair (°C)':'trh', 'm_trh1_RH (%)': 'trh_ref', 'm_trh_RH (%)':'trh'})
        data = data[['bui', 'trh_ref', 'trh']]
        for bui, df in data.groupby('bui'):
            df.dropna(inplace=True)
            X = df['trh'].to_numpy().reshape(-1, 1)
            y = df['trh_ref'].to_numpy().reshape(-1, 1)
            X_train, X_test, y_train, y_test = train_test_split(X, y)

            regr = linear_model.RidgeCV().fit(X, y)
            coefs[(sen, bui)] = regr.coef_[0][0]

            if plot:
                y_pred = regr.predict(X_test)
                fig, ax = plt.subplots(**style.size(0.4))
                ax.set_title(f'Korrekturfaktor {sen} {BUID[bui]}')
                ax.scatter(X_test, y_test, color="black", s=.5, label='beobachtete Werte')
                ax.plot(X_test, y_pred, color="blue", linewidth=1, label='Regressionsgerade')
                text = '\n'.join([
                    f"Mittleres Residuenquadrat: {mean_squared_error(y_test, y_pred):.2n}",
                    f"Bestimmtheitsmaß $R^2$: {r2_score(y_test, y_pred):.2n}",
                    f"Ermittelter Faktor: {regr.coef_[0][0]:.3n}"])
                if sen == 'Tair':
                    ax.set(ylabel=r'Sensor am Netzwerk $[\si{\celsius}]$', xlabel=r'Referenzsensor $[\si{\celsius}]$')
                elif sen == 'RH':
                    ax.set(ylabel=r'Sensor am Netzwerk $[\si{\percent rH}]$', xlabel=r'Referenzsensor $[\si{\percent rH}]$')
                ax.text(0.03,1, text, transform=ax.transAxes, ha='left', va='top')
                ax.grid()
                ax.legend(ncol=2, frameon=False, bbox_to_anchor = (1, .98), loc='lower right')
                fig.tight_layout()
                if toTex:
                    style.toTex('pp', f'LineareKorrekturTRH_{bui}_{sen}')
    faktoren = pd.Series(coefs).unstack().T.rename(columns={'RH':'Rh'}).round(3)
    faktoren.to_csv(r'.\data\KorrekturFaktoren_lin.csv')
    if toTex:
        (faktoren.set_axis([r'rel. Luftfeuchte (rH) [\si{\percent}]', r'Lufttemperatur ($T_{air}$) [\si{\celsius}]'], axis=1)
                    .rename_axis('Haus').rename_axis('Messwert', axis=1)
                    .style
                    .format(escape = 'latex', precision=4, na_rep='MISSING', thousands=" ")
                    .format_index(lambda v: BUID[v])
                    .to_latex(
                        './LaTex/tbls/tbl_pp_Korrekturfaktoren_lin.tex',
                        caption='Lineare Korrekturfaktoren zur behebung des Messfehlers der TRH-Sensoren',
                        clines="skip-last;data",
                        position= '!ht',
                        siunitx=True,
                        hrules=True,
                        convert_css=True,
                        position_float="centering",
                        multicol_align="|c|",
                        label='tbl:trhkorrektur_lin')
                    )
    return pd.read_csv(r'.\data\KorrekturFaktoren_lin.csv', index_col=[0])

def updateKorrektur_fix(plot=True, toTex=False):
    if 'IND' not in globals():
        getIND()
    deltaTair = IND.loc[:,pd.IndexSlice[:,'O', 'SWK',:]].filter(like='Tair').filter(like='trh').groupby(level=0, axis=1).diff(axis=1).dropna(how='all', axis=1).dropna(axis=0).droplevel([1,2,3],axis=1).resample('H').mean()
    deltaRH = IND.loc[:,pd.IndexSlice[:,'O', 'SWK',:]].filter(like='RH').filter(like='trh').groupby(level=0, axis=1).diff(axis=1).dropna(how='all', axis=1).dropna(axis=0).droplevel([1,2,3],axis=1).resample('H').mean()
    
    if plot:
        for bui, temp in deltaTair.iteritems():
            fig, axs = plt.subplots(2,2, sharex='col', sharey='row', gridspec_kw={'width_ratios':[5,1]})
            fig.suptitle(f'Differenz der TRH Sensoren\n{BUID[bui]}')

            dot = temp.index
            mean = temp.rolling('24H').mean()
            error = temp.rolling('24H').std()

            ax = axs[0][0]
            ax.plot(mean, color='#CC4F1B',label = 'deltaT')
            ax.fill_between(dot, np.array(mean-error).flatten(), np.array(mean+error).flatten(), alpha=0.5, edgecolor='#CC4F1B', facecolor='#FF9848', label='stdT')
            ax.legend()
            ax.grid()
            ax.set(ylabel=r'$\Delta K$')
            sns.boxplot(data=deltaTair[bui], ax = axs[0][1], color='#FF9848')
            rh = deltaRH[bui]

            dot = rh.index
            mean = rh.rolling('24H').mean()
            error = rh.rolling('24H').std()
            ax = axs[1][0]
            ax.plot(mean, color='#1B2ACC',label='deltaRH')
            ax.fill_between(dot, np.array(mean-error).flatten(), np.array(mean+error).flatten(), alpha=0.2, edgecolor='#1B2ACC', facecolor='#089FFF', linewidth=4, linestyle='dashdot', antialiased=True,label='std(RH)')
            ax.legend()
            ax.grid()
            ax.set(ylabel=r'$\Delta \si{\percent}-rH $')
            sns.boxplot(data=deltaRH[bui], ax = axs[1][1], color='#089FFF')
            axs[1][1].set_xticks([0], [bui], visible=False)

            ax.xaxis.set_major_locator(mpl.dates.MonthLocator())
            ax.xaxis.set_major_formatter(mpl.dates.DateFormatter("%b '%y"))

            fig.tight_layout(pad=0.1)

    df = pd.DataFrame({'Rh': -deltaRH.median(), 'Tair': -deltaTair.median()}).round(2)
    df.to_csv(r'.\data\KorrekturFaktoren_fix.csv')
    if toTex:
        (df.set_axis([r'rel. Luftfeuchte (rH) [\si{\percent}]', r'Lufttemperatur ($T_{air}$) [\si{\celsius}]'], axis=1)
                    .rename_axis('Haus').rename_axis('Messwert', axis=1)
                    .style
                    .format(escape = 'latex', precision=1, na_rep='MISSING', thousands=" ")
                    .format_index(lambda v: BUID[v])
                    .to_latex(
                        './LaTex/tbls/tbl_pp_Korrekturfaktoren_konst.tex',
                        caption='Konstante Korrekturfaktoren zur behebung des Messfehlers der TRH-Sensoren', 
                        clines="skip-last;data",
                        position= '!ht',
                        siunitx=True, 
                        hrules=True, 
                        convert_css=True,
                        position_float="centering",
                        multicol_align="|c|",
                        label='tbl:trhkorrektur_konst')
                    )
    return pd.read_csv(r'.\data\KorrekturFaktoren_fix.csv', index_col=[0])        

def getEnergymeter(bui, app):
    if 'files' not in globals(): scan_eb_folder()
    EM = pd.read_csv(files['em'][bui]['1min'], index_col = [0], header=[0,1,2],low_memory=False)
    EM.index = pd.to_datetime(EM.index, utc=True).tz_convert('Europe/Berlin')
    EM = EM.drop('TPID',axis=1,level=2).rename(columns={'2OG-Nord' : 'N','2OG-Ost':'O','2OG-Sued':'S'}).dropna(how='all', axis=1)
    EM.columns = pd.MultiIndex.from_tuples([(col[0], '_'.join(col[1:]).strip()) for col in EM.columns.values])
    EM = EM.rename(columns={'H_HQ':'Wärmemenge','H_VW':'Warmwasser','W_VW':'Kaltwasser'})
    for col in EM.filter(like='Wärmemenge').columns:
            EM[col] = EM[col]/1000 #kWh
    EM = EM[pd.MultiIndex.from_product([['N', 'S', 'O'], ['Warmwasser', 'Wärmemenge', 'Kaltwasser']])].copy()
    EM = EM.fillna(method='ffill')
    for col, coldata in EM.iteritems():
            minval = coldata.loc[coldata.first_valid_index()]
            maxval = coldata.loc[coldata.last_valid_index()]
            EM[col] = coldata.where(((coldata > minval) & (coldata < maxval)))
            EM[col] = EM[col] - minval
    return {col: EM[(app, col)].rename((bui, app, 'WE')) for col in EM[app]}

def getAMB(update = False):
    global AMB
    if 'files' not in globals():
        scan_eb_folder()
    
    def getWeatherdata():
        path = files['tf']['WD']['1min']
        df = pd.read_csv(
            path, 
            decimal='.', 
            na_values = '#N/V',
            low_memory=False
            )
        df.set_index( pd.to_datetime(df['Datetime'], utc=True).dt.tz_convert('Europe/Berlin'), inplace=True)
        df.drop('Datetime',axis=1,inplace=True)
        df.columns = ['ID', 'T_amb', 'Rh_amb','windspeed','gustspeed','rain','winddir','btry']
        df.drop(['ID', 'btry'],axis = 1,inplace=True)
        df['rain'].replace(0,np.nan,inplace=True)
        return df

    def getPyranometer():
        path = files['tf']['PM']['1min']
        df = pd.read_csv(
            path, 
            decimal='.', 
            na_values = '#N/V',
            low_memory=False
            )
        df.set_index( pd.to_datetime(df['Datetime'], utc=True).dt.tz_convert('Europe/Berlin'), inplace=True)
        df.drop('Datetime',axis=1,inplace=True)
        df['Direct W/m^2'][df['Direct W/m^2'] < 0] = 0
        df['Diffuse W/m^2'] = df['Global W/m^2'] - df['Direct W/m^2']
        df.columns = df.columns.str.split(' ',expand=True).droplevel(level=1).str.lower()
        return df
    
    if not os.path.isfile('./_db/robustDBs/AMB.pkl'):
        print('Es wurde keine Wetter Datenbank gefunden. Es wird eine neue aus dne Rohdaten erstellt')
        update = True
    if update:
        print('Aktualisiere WetterDB...                                                              ')
        AMB = pd.merge(getWeatherdata(), getPyranometer(), left_index=True, right_index=True)
        #AMB['season'] = AMB.index.to_series().apply(tb.utils.get_season)
        AMB.to_pickle('./_db/robustDBs/AMB.pkl')
    else:
        print('Lade WetterDB...')
        AMB = pd.read_pickle('./_db/robustDBs/AMB.pkl')
    return AMB

### TinkerForge Raumklima DB
def getIND(update = False):
    global IND
    if 'files' not in globals():
        scan_eb_folder()

    def load_tf_bui(bui):
        path = files['tf'][bui]['1min']
        df = pd.read_csv(path, decimal='.', na_values = '#N/V',low_memory=False)
        df.replace([' ','  '],np.NAN,inplace=True)
        df.set_index( pd.to_datetime(df['Datetime'], utc=True).dt.tz_convert('Europe/Berlin'), inplace=True)
        df.drop('Datetime',axis=1,inplace=True)
        idx = []
        for col in df.columns:
            sensor = ' '.join(col.split(' '))
            # Für alle Sensorbezeichnungen die jetzt mit der Gebäude ID Anfangen:
            if sensor.startswith(bui):
                # Entferne Gebäudebezeichnung...
                sensor = sensor.split('_',2)[1:]
                # Setze Stochwerksbezeichnung
                sensor = sensor[1].split('_')
                if sensor[0] == 'Dach':
                    index = ('DA', 'WE', '_'.join(sensor).strip())
                elif len(sensor) < 4:
                    if len(sensor[0]) == 1:
                        index = (sensor[0], 'WE', '_'.join(sensor[1:]).strip())
                    else:
                        index = ('DA', 'WE', '_'.join(sensor).strip())
                elif len(sensor) >= 4:
                    if sensor[0] in ['N', 'S', 'O']:
                        index = (sensor[0], sensor[1], '_'.join(sensor[2:]).strip())
                    elif sensor[0] in ['TH']:
                        index = (sensor[0], 'WE', '_'.join(sensor[1:]).strip())
            else:
                if sensor == '-->Extra-Sensors-->':
                    index = ('','','')
                else:
                    index = ('DA', 'WE', sensor.split('-')[1].split('_', 2)[2].strip())
            idx.append(index)
        df.columns = pd.MultiIndex.from_tuples(idx)
        df.sort_index(axis=1,inplace=True)
        
        return df

    if not os.path.isfile('./_db/robustDBs/IND.pkl'):
        print('Es wurde keine Raumklima Datenbank gefunden. Es wird eine neue aus dne Rohdaten erstellt')
        update = True
    
    if update:
        print('Aktualisiere Raumklima...                                                                    ')
        p_log = setup_logger('IND_logger', f'./data/logs/{dt.datetime.now():%y%m%d}_IND_preprocessing.log', printlog=False)
        p_log.info('Starte aktualisierung der Raumklima Datenbank.')
        IND = pd.concat({bui: load_tf_bui(bui) for bui in BUID},axis=1)
            
        # Benenne Fenster lesbarer um.
        windows = {}
        for col in IND.filter(like='reed').columns:
            bui, app, room, window = col
            _window = [w.strip() for w in window.split()]
            ori = _window[0].split('_')[0]
            s = ''.join([s for s in _window[1].replace('-','') if s.isupper()])
            newcol = (bui, app, room, f'Fenster {tb.utils.KOMPASS[ori]} [{s}]')

            i = 0
            while newcol in windows.values():
                oldcol = list(windows.keys())[list(windows.values()).index(newcol)]
                windows[oldcol] = (bui, app, room, f'Fenster {tb.utils.KOMPASS[ori]} [{s}]-{string.ascii_uppercase[i]}')
                i += 1
                newcol = (bui, app, room, f'Fenster {tb.utils.KOMPASS[ori]} [{s}]-{string.ascii_uppercase[i]}')

            windows[(bui, app, room, window)] = newcol
        windows = {oldcol[-1]: newcol[-1] for oldcol, newcol in windows.items()}
        IND = IND.rename(columns=windows,level=3).replace(['Open', 'Closed'], [1, 0])
        
        # Drop useless sensors...
        p_log.info('Sortiere nicht mehr vorhandene Sensoren aus: "(MH, O, SWK, o_tilt)"')
        IND.drop(('MH', 'O', 'SWK', 'o_tilt'), axis=1, inplace=True)
        p_log.info('Sortiere nicht mehr vorhandene Sensoren aus: "(LB, N, K, w_t_Tair (°C))"')
        IND.drop(('LB', 'N', 'K', 'w_t_Tair (°C)'), axis=1, inplace=True)
        
        # Korrektor Kelvin / Celsius Fehler
        p_log.info('Korrigiere einen Fehler bei dem die pt-Sensor ihre Messwerte in K statt °C aufzeichnen.')
        mcols = IND.columns.get_level_values(3).str.contains('_pt_')
        IND.loc[:,mcols] = IND.loc[:,mcols].where(IND.loc[:,mcols] > -200, IND.loc[:,mcols]+273.15)

        p_log.info('Suche nach Fehlerhaften Sensoren...')
        p_log.info('INFO: Wenn ein Sensor über mehr als einen Tag (1440min) den gleichen Wert ausgibt, wird er als fehlerhaft angenommen und die entsprechenden Werte werden entfernt.')
        p_log.info('INFO: Hiervon ausgenommen sind binäre Sensoren wie die Bewegungsmelder und die Fenster.')
        for col, coldata in IND.iteritems():
            bui, app, room, sensor = col
            if any([x in i for i in col for x in ['md', 'Fenster', 'E-']]): 
                continue
            eva = IND[col].rolling('D').std()
            cond = (eva.abs() < 0.01)
            crossing = (cond != cond.shift()).cumsum()
            count = eva.groupby(crossing).transform('size')
            df = pd.concat([IND[col], eva, cond, crossing, count], keys=['value', 'std', 'cond', 'crossing', 'count'], axis=1)
            df.loc[df.cond == False, 'count'] = 0
            df['count'] = df['count'].where(df['count'] > 1440)
            if df['count'].max() > 0:
                summary = df.reset_index().groupby('crossing').agg({'Datetime':['first', 'last'], 'count':['mean']}).dropna().set_axis(['Start', 'End', 'Duration'], axis=1)
                IND.loc[df['count'] > 0, col] = np.NaN
                for i, row in summary.iterrows():
                    p_log.info(f'{bui}-{app}-{room}: {sensor}: Fehlerhafte Daten von {row["Start"]} bis {row["End"]}')
        p_log.info('Speichere aktualisierte Datenbank ab...')
        IND.to_pickle('_db/robustDBs/IND.pkl')
    else:
        print('Lade Raumklima...')
        IND = pd.read_pickle('./_db/robustDBs/IND.pkl')
    
    return IND

#############################       PREPROCESSING         ###################################
    
def preprocessTRH(bui, app, db=None, sensortype=None, correction='fix', correction_except=['B'], **kwargs):
    if correction is not None:
        if correction == 'fix':
            KORREKTUR = pd.read_csv('./data/KorrekturFaktoren_fix.csv', index_col=[0])
        elif correction == 'lin':
            KORREKTUR = pd.read_csv('./data/KorrekturFaktoren_lin.csv', index_col=[0])
        elif correction == False:
            pass
    value = {'RH (%)': 'Rh', 'Tair (°C)':'Tair'}
    ranges = {'Tair':(-20, 40), 'Rh': (0, 100)}
    if db is None:
        if 'IND' in globals():
            db = IND[bui][app]
        else:
            db = getIND()[bui][app]
    else:
        db = db[bui][app]
    df = db[db.columns[db.columns.get_level_values(-1).str.contains('trh')]]
    df.columns = pd.MultiIndex.from_tuples([(*ids[:-1], *ids[-1].rsplit('_', 1)) for ids in df.columns])
    df = df.rename(columns = value).groupby(level=[0,-1],axis=1).median().round(1)
    dfs = []
    for room, group in df.groupby(level=[*range(df.columns.nlevels-1)], axis=1):
        group = group[room]
        group = pd.concat({room: group.where(group['Rh'].between(*ranges['Rh']) & group['Tair'].between(*ranges['Tair']))}, axis=1)
        if correction:
            for (room, value), col in group.iteritems():
                if room not in correction_except:
                    if correction == 'fix':
                        group.loc[:,(room, value)] = col + KORREKTUR.at[bui, value]
                    elif correction == 'lin':
                        group.loc[:,(room, value)] = col.mul(KORREKTUR.at[bui, value]).round(1)
                    elif correction == False:
                        pass
        dfs.append(group)
    df = pd.concat(dfs, axis=1)

    if sensortype is None:
        return df
    else:
        return df.groupby(level=[1], axis=1).get_group(sensortype)

def getTHK(db=None, drop=True, Wertebereich=(0, 100), **kwargs):
    if db is None:
        if 'IND' in globals():
            db = IND
        else:
            db = getIND() 
    M = []
    drplvl = []
    Wertebereich = (0, 100)
    nlvl = db.columns.nlevels
    for lvl in range(nlvl):
        for key, item in kwargs.items():
            if item in db.columns.get_level_values(lvl):
                if drop: drplvl.append(lvl)
                M.append(np.array(db.columns.get_level_values(lvl).str.contains(item)))
        if db.columns.get_level_values(lvl).str.contains('pt_Thk').any():
            sensor = np.array(db.columns.get_level_values(lvl).str.contains('pt_Thk'))
    df = db[db.columns[np.array(M).all(axis=0) & sensor]]
    df.columns = pd.MultiIndex.from_tuples([(*ids[:-1], *ids[-1].rsplit('_', 1)) for ids in df.columns])
    df = df.rename(columns = {'Thk (°C)':'Thk'}).groupby(level=[*range(0,nlvl-1),-1],axis=1).median().round(1).droplevel(drplvl, axis=1)
    if Wertebereich is not False:
        df = df[(df.gt(Wertebereich[0])) & df.lt(Wertebereich[1])]
    return df


def removeUnoccupied(df):
    # try:
    #     einzugsdaten = pd.read_csv('./data/eb_Einzugsdaten.csv',sep=';', index_col=[0], header=[0,1]).unstack().dropna().reset_index().drop(0,axis=1)
    #     einzugsdaten.columns = ['bui','app','Einzug']
    #     einzugsdaten.set_index(['bui', 'app'],inplace=True)
    #     Einzug = pd.DataFrame((pd.to_datetime(einzugsdaten['Einzug'], format='%d.%m.%Y')).dt.tz_localize('Europe/Berlin'))
    # except FileNotFoundError:
    #     print('Es wurde keine Datei mit Einzugsdaten gefunden. Es werden alle gefundenen Datensätze verwendet.')
    match = {item.name: lvl for lvl, items in enumerate(df.columns.levels) for item in style.STARTDATES.index.levels if (items.isin(item)).any()==True}
    for col, group in df.groupby(level=list(match.values()), axis=1):
        df.loc[:style.STARTDATES.at[col, 'Einzug'], col] = np.NaN
    return df


def preprocessWindowSignals(df):
    s = df.squeeze().copy()
    window = s.name
    # Räume auf und entferne Fehlerhafte Messwerte.
    data = s.rolling('3min').quantile(0.5, interpolation='nearest').where(lambda x: (x.resample('D').mean().asfreq('min', 'ffill') < 1) & s.notna())
    _df = data.fillna(-1).reset_index()
    _df = _df.groupby((_df[window] != _df[window].shift()).cumsum()).agg({'Datetime': ['min', 'max', aggMINMAX], window:'mean'}).set_axis(['Start', 'End', 'Dauer', 'Zustand'], axis=1).where(lambda x: (x.Zustand>0) & (x.Dauer.between(pd.Timedelta(minutes=1), pd.Timedelta(days=1)))).dropna(how='all')

    opening = _df.set_index('Start')['Zustand']
    closing = - _df.set_index('End')['Zustand']
    return opening, closing, data


#############################       SIGNALS         ###################################

def getAnwesenheit(bui, app, db=None, date=None, plot=False, **kwargs):
    if not date and not plot: m = IDX[:]
    elif date: m = date
    elif plot and not date: m = IDX['2022-04-25':'2022-04-30']
    else: Warning('Inputs inconsistent...')
    
    if db is None:
        if 'IND' in globals():
            db = IND
        else:
            db = getIND()

    _df = db.loc[m,(bui, app)]
    
    # Bewegungsmelder
    md = _df.filter(like='_md').max(axis=1).replace(0,np.NaN).dropna()
    #print(f'{BUID[bui]} Wohnung {APPS[app]}: Im Zeitraum vom {_df.index.min():%d.%m.%y} bis {_df.index.max():%d.%m.%y} wurden {len(md)} Bewegungen festgestellt.')
    if len(md) == 0: Warning('Keine Bewegungen festgestellt.')
    
    # 15min vor und nach einer Bewegungsmelderaktivierung
    MD = md.reindex_like(_df).rolling('30T', center=True).max().gt(0)
    
    # Fenster
    _window = {'IO':[], 'data':[]}
    for (room, window), data in _df.filter(like='Fenster').groupby(level=[0,1], axis=1):
        if len(data[room].dropna()) > 0:
            opening, closing, df = preprocessWindowSignals(data[room])
            _window['IO'].append([opening, closing])
            _window['data'].append(df)
    win_open = pd.concat(list(map(list, zip(*_window['IO'])))[0])
    win_close = pd.concat(list(map(list, zip(*_window['IO'])))[1])
    windows = pd.concat(_window['data'], axis=1).max(axis=1)
    if len(windows) == 0:   Warning('Keine Fensterdaten vorhanden.')
    if len(win_open) == 0 or len(win_close) == 0:   Warning('Keine Fensteröffnungen erkannt.')
   
    NAs = _df.filter(regex=r'Fenster|md|co2|trh').notna().groupby(level=0, axis=1).any().any(axis=1)
    # 15min vor und nach einer Fenster-Schließung
    WC = win_close.resample('T').sum().reindex_like(_df).rolling('30T', center=True).max().gt(0)
    # 15min vor und nach einer Fensteröffnung
    WO = win_open.resample('T').sum().reindex_like(_df).rolling('30T', center=True).max().gt(0)
    
    # CO2
    # Unbewohnutes Ost-Appartment als Referenz
    co2_ref = db.loc[m,(bui, 'O')].filter(like='ppm').droplevel(1,axis=1).squeeze().rolling('H').median()
    # Sensoren im Schlafzimmer als Indikator für die Gesamte WE
    co2_app = _df.filter(like='ppm').droplevel(1,axis=1).squeeze()
    if len(co2_app) == 0: Warning('Keine CO2 Daten gefunden')
    # Bedinungung (A)
    # Berechne die Mittelwerte in gleitenden 15min Schritten und berechne die Differenz der Schritte,
    # Betrachte dann den Mittelwert über zwei Zeitstunden; Wenn dieser größer 0 ist, wird die CO2 Konzentration als steigend angenommen.
    CO2_A = co2_app.rolling('15min', center=True).mean().diff().resample('2H').mean().gt(0).reindex_like(_df, method='ffill')
    # Bedinungung (B)
    # Der 15min. rollende Mittelwert muss größer sein, als der geleiche Wert der unbewohnten Referenzwohnung.
    CO2_B = co2_app.rolling('15min', center=True).mean().gt(co2_ref.rolling('15min', center=True).mean() * 1.1).where(windows == 0)
    # Bedinungung (C)
    # Der 24h. rollende Mittelwert muss größer sein, als der geleiche Wert der unbewohnten Referenzwohnung.
    #CO2_C = co2_app.rolling('D',center=True).mean().gt(co2_ref.rolling('D',center=True).mean())
    # Bedinungung (D)
    # Zusätzlich werden die Tagesmittelwerte betrachtet; Das ist relevant, wenn im Sommer mit offenen Fenstern geschlafen wird. Dies führt zu relativ geringen CO2 Peaks, trotz Anwesenheit.
    # Daher wird als Zusätzliches Kriterium unter der Bedinung, dass die Fenster geöffnet sind, der gleitende 15minütige Mittelwert mit dem Tagesmittelwert verglichen.
    CO2_D = co2_app.rolling('15T',center=True).mean().gt(co2_app.rolling('D',center=True).mean()).where(windows > 0)
    # Kummuliere die Datensätze
    # CO2:
    # (Bedinungung A) unter der Bedingung, dass die CO2 Konzentration höher als, dass 1.1x der Referenzkonzentration ist (Bedigung B), soolange die Fenster geschlossen sind.
    # Zusätzlich muss die mittlere CO2-Konzentration über 24h größer sein, als der 24h Mittelwert der Referenzkonzentration (Bedingung C)
    # Wenn die Fenster geöffnet sind, greigt die (Bedingung D). 
    #CO2 = (CO2_A & CO2_B).where(CO2_C) | CO2_D
    CO2 = (CO2_A & CO2_B) | CO2_D
    # (abs.) Luftfeuchte
    _g = pd.concat({(room, 'g_abs'): tb.comf.g_abs(group['Tair'].squeeze(), group['Rh'].squeeze()) for room, group in pd.concat({'Tair': _df.filter(like='trh_Tair').groupby(level=0, axis=1).mean(), 'Rh':_df.filter(like='trh_RH').groupby(level=0, axis=1).mean()}, axis=1).groupby(level=1,axis=1)}, axis=1, sort=True)
    # Betrachte die mittlere Feuchtebilanz pro Stunde:
    g_eval = _g.diff().resample('H').mean().reindex_like(_g, 'ffill')
    # Berechne das quadratische Mittel der Abweichung zwischen den Räumen, wenn diese größer Null ist, gibt es in einem Raum einen externen Feuchteeintrag.
    # Bei einen erkannten externen Feuchteeintrag wird der Raum als Occupiert betrachtet.
    g_eval = g_eval.sub(g_eval.mean(axis=1), axis=0)[[x for x in ['K', 'B'] if x in g_eval.columns.get_level_values(0)]].pow(2).sum(axis=1).rolling('15T', center=True).sum().round(3)
    G = g_eval.gt(0)
    # Summe
    # Zusammenführen und den gleitenden Wert über 60min bilden.
    anw = pd.concat([CO2, MD, WC, WO, G], axis=1).max(axis=1).resample('H').quantile(.5, interpolation='nearest')
    # Resampling auf Stundenschritte. Es wird stets der Median verwendet. Heißt 50% oder mehr Werte einer Stunde müssen Anwesenheit anzeigen, damit eine Stunde als anwesend Klassifiziert wird.
    anw = anw.asfreq('T', 'ffill').rename('Anwesenheit')
    # Annahme: Wer einmal Nachts zuhause ist geht erst am nächsten Morgen wieder aus dem Haus
    anw_clean = anw.between_time('23:00','07:00').asfreq('T').rolling('3H',center=False).max().fillna(anw).where(NAs)

    if plot:
        _size = style.size(aspect=0.3)
        figs, axs = [], []

        fig, ax = plt.subplots(**_size)
        ax.plot(anw_clean, alpha=0.5, color=style.red, label=r'Anwesenheit')
        ax.fill_between(anw_clean.index, anw_clean, color=style.red, alpha=0.5)
        ax.set(yticks=[0,1], yticklabels= [0,1], ylabel=r'Anwesenheit', xlim=(anw.index.min(), anw.index.max()))
        ax.legend(ncol=5, bbox_to_anchor=(1,1), loc='lower right')
        fig.tight_layout()
        
        figs.append(fig)
        axs.append(ax)

        fig, ax = plt.subplots(**_size)
        ax.vlines(win_open.index, ymin=0, ymax=1, label = r'F. Öffnung', lw=1, color=style.red)
        ax.vlines(win_close.index, ymin=0, ymax=1, label = r'F. Schließung', lw=1, color=style.blue)
        ax.fill_between(windows.index, 0, 1, where=(windows > 0), alpha=0.25, label = r'F. geöffnet', transform=ax.get_xaxis_transform())
        ax.vlines(md.index, ymin=0, ymax=1, ls='solid', label = r'Bewegungsmelder', lw=1, color=style.yellow)
        ax.set(yticks=[0,1], yticklabels = [0,1], ylabel='[I/O]', xlim=(anw.index.min(), anw.index.max()))
        ax.legend(ncol=5, bbox_to_anchor=(1,1), loc='lower right')
        fig.tight_layout()
        
        figs.append(fig)
        axs.append(ax)

        fig, ax = plt.subplots(**_size)
        ax.plot(co2_app, color=style.Greens(255), label = r'SZ')
        ax.plot(co2_ref, color=style.Greens(175), label = r'ref. Whg.', ls='dashed')
        ax.fill_between(CO2.index, 1, where=(CO2 > 0), alpha=0.25, label =r'abgl. Kriterium', color=style.Greens(100), transform=ax.get_xaxis_transform())
        ax.set(ylabel=r'[ppm]', xlim=(anw.index.min(), anw.index.max()))
        ax.legend(ncol=5, bbox_to_anchor=(1,1), loc='lower right')
        fig.tight_layout()
        
        figs.append(fig)
        axs.append(ax)

        fig, ax = plt.subplots(**_size)
        ax.plot(_g, lw=0.5)
        ax.fill_between(G.index, 1, where=G>0, alpha=0.25, label = r'abgl. Kriterium', transform=ax.get_xaxis_transform())
        ax.set(ylabel=r'[\si{\gram\per\kilogram}]', xlim=(anw.index.min(), anw.index.max()), xlabel='')
        ax.legend(ncol=5, bbox_to_anchor=(1,1), loc='lower right')
        fig.tight_layout()
        figs.append(fig)
        axs.append(ax)

        for _ax in axs:
            _ax.grid(axis='x', which='minor')
            _ax.grid(axis='x', which='major')
            _ax.xaxis.set_minor_locator(mpl.dates.HourLocator([0,6,12,18,24]))
            _ax.xaxis.set_major_locator(mpl.dates.DayLocator(interval=1))

        if 'export' in kwargs:
            if kwargs['export']:
                style.toTex('pp', 'anw_results', figs[0])
                style.toTex('pp', 'anw_fenster', figs[1])
                style.toTex('pp', 'anw_co2', figs[2])
                style.toTex('pp', 'anw_gabs', figs[3])
    return anw_clean

def getHKIO(bui, app, db=None, Fenster=None, plot=False, correction='fix', **plot_kwargs):
    if db is None:
        if 'IND' not in globals():
            db = getIND()
        else:
            db = IND

    # Extrahiere Raumtemperatur
    _Tair = preprocessTRH(bui, app,db=db, sensortype='Tair', correction=False)

    # Extrahiere Heizkörpertemperatur
    _Thk = getTHK(db, bui=bui, app=app, drop=True)

    # Extrahiere Fensterstatus
    if not isinstance(Fenster, (pd.Series, pd.DataFrame)):
        #_Fenster = pd.concat({'Fenster': db[bui][app].filter(like='Fenster').groupby(level=[0], axis=1).sum(min_count=1)}, axis=1).swaplevel(axis=1)
        Fenster = getFensterDatenbank(bui, app, IO=False)
    # Klassifizierung Heizung An/Aus
    _hk = pd.concat([_Thk,_Tair, Fenster], axis=1).dropna(how='all').swaplevel(axis=1)
    hkio = (_hk['Tair'][_hk['Thk'].columns].resample('15T').quantile(0.5, interpolation='nearest').apply(np.ceil) < _hk['Thk'].resample('15T').quantile(0.25, interpolation='nearest').apply(np.floor)).resample('H').max().asfreq('T', 'ffill').astype(int).where(_Thk.droplevel(1, axis=1).notna())
    _hk = pd.concat([_hk, pd.concat({'HK_IO': hkio}, axis=1)], axis=1).swaplevel(axis=1).sort_index(axis=1)
    
    # Bestimmung Tset
    for r, (room, group) in enumerate(_hk.groupby(level=0, axis=1)):
        if 'Thk' in group[room].columns: 
            _df = group[room].assign(Tair_m24 = lambda df: np.ceil(df.Tair.rolling('D', center=True).median()).where(df.Thk.notna())).reset_index()
            _df = _df.groupby([((_df['HK_IO'] != _df['HK_IO'].shift()) & _df['HK_IO'].notna()).cumsum()]).agg({'Datetime': ['min', 'max', aggMINMAX], 'Tair': ['min', 'max', getTset], 'Thk': ['mean', 'std', 'median', 'max'], 'HK_IO': ['median'], 'Tair_m24':['median']})
            _df = _df.where(_df[('HK_IO', 'median')] > 0).dropna()
            _df = _df.where(lambda df: df[('Tair_m24', 'median')] < df[('Thk', 'max')].round(0)).dropna()
            x0 = _df[('Datetime', 'min')]
            x1 = _df[('Datetime', 'max')]
            Tset = _df[('Tair', 'getTset')]
            if len(Tset) > 0:
                _hk[(room, 'Tset')] = pd.concat([pd.Series(value, pd.date_range(start, end, freq='T')) for start, end, value in zip(x0, x1, Tset)]).where(_Thk[room].squeeze().notna())
                _hk.loc[:,(room, 'HK_IO')] = _hk[(room, 'HK_IO')].mask((_hk[(room, 'HK_IO')].eq(1) & _hk[(room, 'Tset')].isna()), 0)

    # Bereinungung: Entferne Fehlerhafte Klassifizierung, wenn Fenster geöffnet werden. 
    mean = _hk.loc[:,IDX[:,'HK_IO']].resample('D').mean()           # Betrachte die Tagesmittlewerte des HK-Status
    for room, group in mean.groupby(level=0, axis=1):
        group = group.squeeze()
        outliners = mean.loc[((group < 0.2) & (group > 0) & (group.shift(-1) < 0.2) & (group.shift(1) < 0.2)), room]
        for item in outliners.index:
            idx = pd.date_range(item, periods = 1440, freq='T')
            df = _hk.loc[idx , room]
            if all(x in df.columns for x in ['Tset', 'Tair', 'HK_IO', 'Thk', 'Fenster']):
                _Tair_day = df['Tair'].resample('D').median().reindex(idx, method='ffill')
                _Thk_ = df['Thk'].resample('H').max().reindex(idx, method='ffill')
                _hk.loc[idx,(room, 'HK_IO')] = _hk.loc[idx,(room, 'HK_IO')].where(_Thk_ > _Tair_day,0)
                _hk.loc[idx,(room, 'Tset')] = _hk.loc[idx,(room, 'Tset')].where(_Thk_ > _Tair_day)

    _hk = _hk.sort_index(axis=1)
    
    if plot:
        fig, axs = plt.subplots(2,1)
        fig.suptitle('Häufigkeit Heizkörpereinstellung')
        sns.histplot(_hk.loc[:,IDX[:,'Tset']].droplevel(1,axis=1), discrete=True, multiple='dodge', ax = axs[0])
        sns.boxplot(data=_hk.loc[:,IDX[:,'Tset']].droplevel(1,axis=1), ax=axs[1])

        if 'start' in plot_kwargs: 
            start = plot_kwargs['start']
            del plot_kwargs['start']
        else:
            start = _hk.index.max()

        if 'end' in plot_kwargs: 
            end = plot_kwargs['end']
            del plot_kwargs['end']
        else:
            end = _hk.index.max()

        if 'export' in plot_kwargs:
            export = plot_kwargs['export']
            del plot_kwargs['export']
        else:
            export = False

        for room, group in _hk.loc[start:end].groupby(level=[0], axis=1):
            group = group[room]
            if 'Thk' in group.columns:
                fig, ax = plt.subplots(figsize=style.set_size('thesis', aspect=0.3))
                ax.set_title(ROOMS[room], loc='left', y=1)
                ax.set(ylabel = r'$[\si{\celsius}]$', ylim=(10,60))
                ax.plot(group['Tair'], label=r'$T_{air}$')
                ax.plot(group['Thk'], label=r'$T_{sur, HK}$')
                try: 
                    ax.plot(group['Tset'], label = r'$T_{set, est.}$', zorder=10)
                except KeyError: 
                    pass
                if 'Fenster' in group.columns: 
                    ax.fill_between(group.index, 0, 1, where=group['Fenster'] > 0, alpha=0.33, label=r'$Fenster_{IO}$', transform=ax.get_xaxis_transform())
                ax.fill_between(group.index, 0,1, where=group['HK_IO']>0, alpha=0.5, ls='None', label=r'$HK_{IO}$', transform=ax.get_xaxis_transform())
                ax.xaxis.set_minor_locator(mpl.dates.HourLocator([0,6,12,18,24]))
                ax.xaxis.set_major_locator(mpl.dates.DayLocator(interval=1))
                ax.legend(ncol=4, bbox_to_anchor=(1,1), loc='lower right')
                ax.grid(axis='x',which='both')
                fig.tight_layout()
                if export:
                    style.toTex('pp',f'HK_Klassifikation_{room}', fig)
    return _hk

#############################       ERSTELLE DATENBANK         ###################################

def getDB(update = False, correction = 'fix'):
    global DB
    if 'AMB' not in globals(): getAMB()
    if 'IND' not in globals(): getIND()
    if not os.path.isfile('./_db/robustDBs/DB.pkl'):
        print('Keine Datenbank gefunden. Eine neue wird erstetllt.')
        update = True
    if update:
        print('Aktualisiere Datenbank...')      
        _amb = AMB
        _ind = IND.rename(columns={'F':'WZ'})
        dfs={}
        for b, bui in enumerate(BUID):
            for a, app in enumerate(APPS):
                t0 = time.time()
                i = (3*b+a)+1
                print(f'#{i}/9: {BUID[bui]} {APPS[app]}')
                DF = []         # Leere Liste zum Sammeln der Spalten
                DATA = _ind[bui][app]    # extrahiere Messdaten für entsprechende Wohnung
                Rooms = DATA.columns.get_level_values(0).unique().to_list() # ermittle Räume in aktueller Wohnung
                Rooms.remove('WE')
                # Extrahiere Raumtemperatur
                _Tair = preprocessTRH(db=_ind, bui=bui, app=app, sensortype='Tair', correction=False)
                # Extrahiere Luftfeuchte
                _Rh = preprocessTRH(db=_ind, bui=bui, app=app, sensortype='Rh', correction = correction)
                DF.append(_Rh)
                DF.append(pd.concat({'g_abs': tb.comf.g_abs(_Tair.droplevel(1, axis=1), _Rh.droplevel(1, axis=1))}, axis=1).swaplevel(axis=1))
                # Extrahiere CO2-Konzentration
                DF.append(pd.concat({'CO2': DATA.filter(like=r'co2 (ppm)').groupby(level=[0],axis=1).mean()},axis=1).swaplevel(axis=1))
                # Extrahiere Schwarzkugel Temperatur
                _Tsk = pd.concat({'Tsk': DATA.filter(regex=r'pt_Tsk').groupby(level=[0],axis=1).median()}, axis=1).swaplevel(axis=1)
                DF.append(_Tsk)
                # Extrahiere operative Temperatur
                DF.append(pd.concat({'Top': tb.comf.calcTOP(_Tair.droplevel(1, axis=1), _Tsk.droplevel(1, axis=1))}, axis=1).swaplevel(axis=1))
                # Extrahiere Außenklima
                DF.append(pd.concat([pd.concat({(room): _amb[['T_amb', 'Rh_amb']] for room in Rooms}, axis=1), pd.concat({(room, 'T_amb_g24'): AMB['T_amb'].rolling('24h').mean().round(2) for room in Rooms}, axis=1)], axis=1, sort=True))
                print(f'|----> Messdaten geladen! (1/5)'+100*' ', end='\r')                
                # Extrahiere Fenster...
                Fenster = {}    # sammle alle Fenster in einem dict
                for (room, window), group in DATA.filter(like = 'Fenster').groupby(level=[0,1], axis=1):
                    # Lege für jeden Raum eine Liste an...
                    if room not in Fenster: Fenster[room] = {'IO':[], 'data':[]} 
                    if len(group[room].dropna()) > 0:
                        opening, closing, _df = preprocessWindowSignals(group[room])
                        Fenster[room]['IO'].append([opening, closing])  # Sammle Öffnungen und Schließungen
                        Fenster[room]['data'].append(_df)               # Sammle kontinuierlichen Fensterstatus
                # Forme Fensterdaten um.
                _Fenster = []
                _FensterIO = []
                for room, data in Fenster.items():
                    _Fenster.append(pd.concat({room: pd.concat(data['data'], axis=1)}, axis=1))
                    _FensterIO.append(pd.concat({('Open',room): pd.concat([item[0] for item in data['IO']]).sort_index().resample('T').sum(min_count=1),('Close',room): pd.concat([item[1] for item in data['IO']]).sort_index().resample('T').sum(min_count=1)},axis=1))
                # Sammle einzelne Fenster
                DF.append(pd.concat(_Fenster, axis=1))
                # summierte Fenster pro Raum (wird für HK-Klassifikation gebraucht...)
                _Fenster_sum = pd.concat({'Fenster': (pd.concat(_Fenster, axis=1).groupby(level=0, axis=1).sum(min_count=1))}, axis=1).swaplevel(axis=1)
                # Sammle Fensteröffnungen...
                DF.append(pd.concat({'Fensteröffnung': pd.concat(_FensterIO, axis=1).groupby(level=[1], axis=1).sum(min_count=1)}, axis=1).swaplevel(axis=1))
                print(f'|----> Fenster geladen! (2/5)'+100*' ', end='\r')
                # Berechne Anweseheit
                DF.append(pd.concat({('WE', 'Anwesenheit'): getAnwesenheit(bui, app, db=_ind)}, axis=1))
                print(f'|----> Anwesenheit geladen! (3/5)'+100*' ', end='\r')
                # Extrahiere Wärmemengenzähler
                DF.append(pd.concat({'WE':pd.concat(getEnergymeter(bui, app),axis=1)},axis=1, sort=True))
                print(f'|----> Wärmemengenzähler geladen! (4/5)'+100*' ', end='\r')
                # Berechne Heizkörper Einstellungen
                DF.append(getHKIO(db = _ind, bui = bui, app = app, Fenster = _Fenster_sum, correction=correction))
                print(f'|----> Heizung geladen! (5/5)'+100*' ', end='\r')
                dfs[(bui, app)] = pd.concat(DF, axis=1)
                print(f'|----> Fertig! Dauer: {int(time.time() - t0)} sec.')
        DB = pd.concat(dfs,axis=1).round(1)

        ### Korrekte Einzugsdaten
        DB = removeUnoccupied(DB)
            
        # Setze Spalten / Zeilen Namen
        DB.index = DB.index.set_names('Datetime')
        DB.columns = DB.columns.set_names(['bui', 'app', 'room', 'value'])
        DB.sort_index(axis=1, inplace=True)
        DB.sort_index(axis=0, inplace=True)
        DB.to_pickle('./_db/robustDBs/DB.pkl')

    else:
        print('Lade gespeicherte Datenbank...                                                                ')
        DB = pd.read_pickle('./_db/robustDBs/DB.pkl')
    print('Laden vollständig...                                                                        ', end='\r')
    return DB

# ===== ===== ===== Thermischer Komfort ===== ===== =====
def getAdaptiveComfort(occupied=True, flat=False):
    if 'DB' not in globals():
        getDB()
    Tamb = DB.loc[:,IDX[:,:,:,'T_amb']].droplevel(3, axis=1).groupby(level=[0,1], axis=1).mean()
    Top = estimateOperativeTemperature(mode='fillgaps').droplevel(3, axis=1).groupby(level=[0,1], axis=1).mean()
    if flat:
        KH = pd.concat([tb.comf.KelvinstundenEN(Tamb=Tamb, Temp=Top).rename(columns={'ÜTGS':'ÜTGS_EN', 'UTGS':'UTGS_EN'}),tb.comf.KelvinstundenNA(Tamb=Tamb, Temp=Top).rename(columns={'ÜTGS':'ÜTGS_NA', 'UTGS':'UTGS_NA'})], axis=1).sort_index().rename_axis(['bui', 'app', 'value'], axis=1).sort_index().sort_index(axis=1)
        if occupied:
            KH = KH.stack([0,1]).where(getData(value='Anwesenheit').groupby(level=[0,1], axis=1).median().stack([0,1]) == 1).unstack([1,2]).reorder_levels(['bui', 'app', 'value'], axis=1).sort_index(axis=1)
    else:
        KH = pd.concat({'EN': tb.comf.KelvinstundenEN(Tamb=Tamb, Temp=Top), 'NA': tb.comf.KelvinstundenNA(Tamb=Tamb, Temp=Top)}, axis=1).reorder_levels([1,2,0,3], axis=1).sort_index().rename_axis(['bui', 'app', 'norm', 'value'], axis=1).sort_index().sort_index(axis=1).sort_index()
        if occupied:
            KH = KH.stack([0,1]).where(getData(value='Anwesenheit').groupby(level=[0,1], axis=1).median().stack([0,1]) == 1).unstack([1,2]).reorder_levels(['bui', 'app', 'norm', 'value'], axis=1).sort_index(axis=1).sort_index()
    return KH.asfreq('H')

def calculatePMVPPD(update=False, **kwargs):
    if 'DB' not in globals():
        getDB()

    if os.path.isfile('./_db/robustDBs/PMV_PPD_params.pkl'):
        params = pd.read_pickle('./_db/robustDBs/PMV_PPD_params.pkl').to_dict()
    else:
        params = {'vair': 0.15,'clo_winter': 1.0,'clo_summer': 0.5, 'met_SZ': 1.0, 'met_rest': 1.2}

    for key, item in kwargs.items():
        if key in params:
            if params[key] != item:
                params[key] = item
                update = True
        else:
            print(f'{key} wurde nicht korrekt definiert - Standardwert wid verwendet. Erlaubte Parameter sind: {list(params.keys())}')

    if not os.path.isfile('./_db/robustDBs/PMV_PPD.pkl'):
        update = True
    if update: 
        print('Berechne PMV und PPD für Schlafzimmer und Wohnzimmer')
        _db = pd.concat([DB.filter(regex=r'Tair|Rh[^_]'), estimateOperativeTemperature(mode='fillgaps')], axis=1).sort_index(axis=1)
        dfs = {}
        for (bui, app, room), group in _db.groupby(level=[0,1,2], axis=1): 
            if room not in ['WZ', 'SZ', 'SWK']:
                continue
            try: 
                df = group[bui][app][room][['Tair','Top', 'Rh']]
                df = df.resample('H').median()
                df = df.dropna()
            except KeyError:
                continue

            if room == 'SZ':
                df['met'] = params['met_SZ']     # schlafen: 0.8, ruhig sitzend: 1.0, leichte Arbeit im Sitzen: 1.2, stehen: 1.4
            else:
                df['met'] = params['met_rest']
            
            df['clo'] = params['clo_winter']     # Sommer: 0.5, Winter: 1.0, Shorts+TShirt: 0.3, nackt: 0
            
            # Nach Monaten
            df['clo'].mask(df.index.month.isin(range(3,9)), params['clo_summer'])

            # Luftgeschwindigkeit
            df['vair'] = params['vair']    # 

            df['v_rel'] = df.apply(lambda x: float(v_relative(x['vair'], x['met'])), axis=1)
            df['clo_d'] = df.apply(lambda x: float(clo_dynamic(x['clo'], x['met'])), axis=1)

            results = pmv_ppd(tdb = df['Tair'], tr = df['Top'], vr = df['v_rel'], rh = df['Rh'], met = df['met'], clo = df['clo_d'], standard="ashrae")
            
            df = pd.concat([df, pd.DataFrame(results, index=df.index)], axis=1)
            dfs[(bui, app, room)] = df
        ThermalComfort = pd.concat(dfs,axis=1).rename_axis(['bui', 'app', 'room', 'value'], axis=1).asfreq('H')
        pd.Series(params).to_pickle('./_db/robustDBs/PMV_PPD_params.pkl')
        ThermalComfort.to_pickle('./_db/robustDBs/PMV_PPD.pkl')
    else:
        print('Öffne PMV-PPD Datenbank...')
        ThermalComfort = pd.read_pickle('./_db/robustDBs/PMV_PPD.pkl')
        params = pd.read_pickle('./_db/robustDBs/PMV_PPD_params.pkl')
    return ThermalComfort, params

# ===== ===== ===== LÜFTUNG ===== ===== ===== =====
def createLüftungsDB(update=False):
    global LüftungsDB
    if 'DB' not in globals():
        getDB()
    if not os.path.isfile('./_db/robustDBs/LüftungsDB_A.pkl'):
        update = True
    if update:
        print('Aktualisiere LüftungsDB...')
        dfs = {}
        for roomID, group in DB.groupby(level=[0,1,2], axis=1):
            group = group[roomID]
            if all(x in group.columns for x in ['HK_IO', 'Thk', 'Top', 'Tset', 'Fenster']):
                group = group.assign(Datetime = lambda df: df.index)
                infos = ['first', 'last', 'min', 'max', 'median', 'mean']
                m = group['Fenster'].dropna()
                if 'CO2' in group.columns:
                    df = group.groupby((m != m.shift()).cumsum()).agg({
                        'Datetime': ['first', 'last',tb.utils.aggMINMAX], 
                        'Fenster': 'mean',
                        'Tair': infos,
                        'Rh': infos, 
                        'Top': infos, 
                        'CO2': infos,
                        'HK_IO': 'mean', 
                        'Tset': infos,
                        'Thk': infos,
                        'T_amb' :infos, 
                        'T_amb_g24': infos,    
                        'Rh_amb': infos, 
                        }).round(1)
                else:
                    df = group.groupby((m != m.shift()).cumsum()).agg({
                        'Datetime': ['first', 'last', tb.utils.aggMINMAX], 
                        'Fenster': 'mean',
                        'Tair': infos,
                        'Rh': infos, 
                        'Top': infos, 
                        'HK_IO': 'mean', 
                        'Tset': infos,
                        'Thk': infos,
                        'T_amb' :infos, 
                        'T_amb_g24': infos,    
                        'Rh_amb': infos, 
                        }).round(1) 
                if len(df) > 0:
                        dfs[roomID] = df
        LüftungsDB = pd.concat(dfs).rename(columns={'aggMINMAX':'timedelta'})
        LüftungsDB.index.set_names(['bui', 'app', 'room', 'id'], inplace=True)
        LüftungsDB.to_pickle('./_db/robustDBs/LüftungsDB_A.pkl')
    else:
        print('Lade gespeicherte LüftungsDB...')
        LüftungsDB = pd.read_pickle('./_db/robustDBs/LüftungsDB_A.pkl')
    return LüftungsDB

def getLüftungen():
    global DB_Lüftung
    if 'LüftungsDB' not in globals():
        createLüftungsDB()
    print('Extrahiere Lüftungen...')
    DB_Lüftung = {}
    for roomid, group in LüftungsDB.groupby(level=[0,1,2]):
        bui, app, room = roomid
        group = group.droplevel(level=[0,1,2])
        _open = False
        for id, row in group.iterrows():
            if row[('Fenster', 'mean')] > 0:
                if _open:
                    continue
                _open = True
                id_open = id
                opening_time = group.loc[id_open,('Datetime', 'first')]
                print(f'{bui} {app} {room} {opening_time} | Fenster geöffnet                                                                                          ', end='\r')
                continue
            if row[('Fenster', 'mean')] == 0:
                if _open:
                    _open = False
                    id_close = id
                    closing_time = group.loc[id-1,('Datetime', 'last')]
                    _df = group.loc[id_open-1:id_close]
                    if (opening_time == _df.iloc[1]['Datetime']['first']) & (closing_time == _df.iloc[-2]['Datetime']['last']):
                        if bui not in DB_Lüftung: DB_Lüftung[bui] = {}
                        if app not in DB_Lüftung[bui]: DB_Lüftung[bui][app] = {}
                        if room not in DB_Lüftung[bui][app]: DB_Lüftung[bui][app][room] = []
                        DB_Lüftung[bui][app][room].append(_df)

                        duration = closing_time - opening_time
                        
                        print(f'{bui} {app} {room} {opening_time} | Fenster geschlossen                                                                                          ', end='\r')
                        print(f'{bui} {app} {room} {opening_time} | {duration}                                                                                          ', end='\r')
    print('Lüftungen extrahiert...                                                                                          ', end='\r')
    return DB_Lüftung

def extractLüftungfromDB(bui, app, _df):
    if 'DB' not in globals():
        getDB()

    opening_time = _df.iloc[1]['Datetime']['first']
    closing_time = _df.iloc[-2]['Datetime']['last']

    duration = closing_time - opening_time

    _lower = max(_df.iloc[0]['Datetime']['first'], opening_time - (3*duration))
    _upper = min(_df.iloc[-1]['Datetime']['last'], closing_time+3*duration)

    df = DB.loc[_lower: _upper, (bui, app)].swaplevel(axis=1).sort_index(axis=1)
    for col in df.columns[df.isna().all()]:
        if 'Tair' in col:
            raise KeyError(f'{col[0]} im {col[1]} wurde nicht aufgezeichnet. Datensatz kann nicht ausgewertet werden.')
    return df.dropna(how='all').dropna(how='all', axis=1)

def getLüftungsDatasets(bui, app, room, _id=None, batch=False):
    # Lade Lüftungsdatensatz
    if 'DB_Lüftung' not in globals():
        getLüftungen()

    try:
        df = DB_Lüftung[bui][app][room]
    except KeyError as e:
        raise IndexError(f'{e} in dieser Wohnung nicht vorhanden...')
    if not batch:
        found = False
        if _id:
            try:    
                _df = extractLüftungfromDB(bui, app, df[_id])
                df = df[_id]
                found = True
            except KeyError:
                print('Zu dieser Lüftung sind nicht ausreichend Daten vorhanden. Bitte Wähle eine andere...')
        while not found:
            _id = int(input(f'{len(df)} Lüftungsvorgänge gefunden. Welcher soll betrachtet werden?'))-1
            # Lade Messdaten im Lüftungszeitraum
            try:    
                _df = extractLüftungfromDB(bui, app, df[_id])
                df = df[_id]
                found = True
            except KeyError:
                print('Zu dieser Lüftung sind nicht ausreichend Daten vorhanden. Bitte Wähle eine andere...')
    if batch:
        if _id == None:
            raise TypeError('Keine _id übergeben. Im Batch-Modus muss eine _id übergeben werden.')
        else:
            _df = extractLüftungfromDB(bui, app, df[_id])
            df = df[_id]

    # Bereite Datensatz vor...
    df_airnode = _df.swaplevel(axis=1).sort_index(axis=1)[room]
    df_airnode['Fenster_gesamt'] = _df['Fenster'].drop(room, axis=1).sum(axis=1)
    df_airnode.sort_index(axis=1, inplace=True)

    # Trenne Zeiträume vor, nach und während der Lüftung
    before = df.iloc[0]
    during = df.where(df[('Fenster','mean')] > 0).dropna(how='all')
    after = df.iloc[-1]

    # Berechne Kenngößen
    data = pd.Series(dtype='object')
    data['duration'] = during['Datetime']['timedelta'].to_list()[0]
    # Überspringe Lüftungen die kürzer als 3 Minuten sind.
    if data['duration'] < pd.Timedelta(2, 'min'):
        raise KeyError(f'{bui}-{app}-{room}#{_id}: Lüftungsdauer mit {data["duration"]} zu kurz - Datensatz wird kann nicht ausgewertet werden.')

    data['median_after'] = after['Tair']['median']
    data['median_before'] = before['Tair']['median']
    data['median_delta'] = before['Tair']['median'] - after['Tair']['median']
    data['idx_opening'] = during['Datetime']['first'].to_list()[0]
    data['idx_closing'] = during['Datetime']['last'].to_list()[0]
    try:
        data['idx_median_after'] = DB.loc[data['idx_closing']:,(bui, app, room, 'Tair')].where(DB.loc[data['idx_closing']:,(bui, app, room, 'Tair')] > data['median_after']).first_valid_index()
    except TypeError:
        data['idx_median_after'] = after['Datetime']['last']
    data['Topen'] = during['Tair']['first'].to_list()[0]
    data['T_amb_open'] = during['T_amb']['first'].to_list()[0]
    data['Tclose'] = during['Tair']['last'].to_list()[0]
    data['T_amb_close'] = during['T_amb']['last'].to_list()[0]
    data['Tmin'] = during['Tair']['min'].to_list()[0]
    data['idx_Tmin'] = df_airnode.loc[data['idx_opening']:, 'Tair'].idxmin()
    data['Tmax'] = during['Tair']['max'].to_list()[0]
    data['idx_Tmax'] = df_airnode.loc[data['idx_opening']:, 'Tair'].idxmax()

    # Ermittle stabilisierte Raumlufttemperatur nach dem Lüften.
    _df = DB.loc[data['idx_closing']:data['idx_median_after'],(bui, app, room, 'Tair')].rolling('60min').mean().resample('5min').last()
    __df = (_df - _df.shift(-1)).loc[data['idx_closing']:]
    data['idx_stable'] = (__df.where(__df.abs() < 0.01).first_valid_index())

    if not isinstance(data['idx_stable'], pd.Timestamp):
        data['idx_stable'] = np.NaN
        data['Tstable'] = np.NaN
    else:
        data['Tstable'] = _df.loc[data['idx_stable']]
    if not batch:
        print(f"Dauer {data['duration']}, Temperaturpotential {(data['Topen'] - data['T_amb_open']):.1f}")
        print(f"Abkühlzeit: {data['idx_Tmin'] - data['idx_opening']}")
        print(f"Abkühlzeit: {data['idx_Tmin'] - data['idx_opening']}")
        print(f"delta T: {(data['Tmin'] - data['median_before']):.1f} K")
        if data['idx_Tmin'] < data['idx_closing']:
            print("Die Raumtemperatur steigt bereits bei geöffnetem Fenster.")
            print(f"Aufwärmzeit (Fenster offen): {data['idx_closing'] - data['idx_Tmin']}")
        print(f"Aufwärmzeit (Fenster geschlossen): {data['idx_median_after'] - data['idx_closing']}")
        print(f"delta T: {(data['median_after'] - data['Tmin']):.1f} K")
    return _id, df_airnode, df, data

def getLüftungsStats(update=False):
    if not update and os.path.isfile('./_db/robustDBs/LüftungsDB_B.pkl'):
        df = pd.read_pickle('./_db/robustDBs/LüftungsDB_B.pkl')
    else:
        dfs = {}
        for bui in BUID:
            for app in APPS:
                for room in ROOMS:
                    i = 0
                    errors = 0
                    stop = False
                    while not stop:
                        try:
                            bui, app, room, _id, df_airnode, df_lüftung, overview = getLüftungsDatasets(bui, app, room, batch=True, _id=i)
                            overview.name = (bui, app, room)
                            dfs[(bui, app, room, _id)] = overview
                            i+=1
                        except KeyError as e:
                            errors += 1
                            print(f'{errors}:{e}', end='\r')
                            i+=1
                        except IndexError:
                            stop = True
                    if i > 0:
                        print(f'Analyse von {bui}-{app}-{room} abgeschlossen: {errors} Fehler in {i} Datensätzen aufgetreten.')
        df = pd.concat(dfs, axis=1).rename_axis(['bui', 'app', 'room', 'id'], axis=1).T
        df['TD_fall_open'] = (pd.to_datetime(df.idx_Tmin) - pd.to_datetime(df.idx_opening))
        df['TD_rise_open'] = (pd.to_datetime(df.idx_closing) - pd.to_datetime(df.idx_Tmin))
        df.sort_index(axis=1, inplace=True)
        df.head()
        df.to_pickle('./_db/robustDBs/LüftungsDB_B.pkl')
    return df


def plotMissingData(df, ori='v', freq='D', level=None, title=None, ax=None, cmap='coolwarm_r'):

    if level is not None:
        df = df.droplevel([*range(level)], axis=1)

    df = df.notnull().resample('D').mean()

    z = df.values

    if cmap is None:
        g = np.zeros((df.shape[0], df.shape[1], 3), dtype=np.float32)
        g[z < 0.5] = [1, 1, 1]
        g[z > 0.5] = style.clrs[2]
    else:
        g = np.zeros((df.shape[0], df.shape[1], 3), dtype=np.float32)
        g = mpl.cm.get_cmap(cmap)(z)

    date_lims = mpl.dates.date2num([df.index.min(), df.index.max()])
    n_sensors = df.shape[1]
    
    if ax is None:
        fig, ax = plt.subplots()

    if title is not None:
        ax.set_title(title)
    if ori == 'v':
        g = np.swapaxes(g, 0, 1)
        labels = [*reversed(list(df.columns))]
        _shape = [date_lims[0], date_lims[1],  0, n_sensors]
        ax.xaxis_date()
        dateaxis = ax.xaxis
        sensoraxis = ax.yaxis
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(True)
        ax.grid(axis='y', which='minor', lw=.5, ls='solid', color=[1,1,1], zorder=3)
        #ax.grid(axis='x', which='major', lw=.5, ls='dashed', color=[1,1,1], zorder=4)
    elif ori == 'h':
        labels = list(df.columns)
        _shape = [0, n_sensors, date_lims[0], date_lims[1]]
        ax.yaxis_date()
        dateaxis = ax.yaxis
        sensoraxis = ax.xaxis
        ax.spines['top'].set_visible(True)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        #ax.grid(axis='x', which='minor', lw=1, ls='solid', color=[1,1,1], zorder=3)
        ax.grid(axis='y', which='major', lw=.5, ls='dashed', color=[1,1,1], zorder=4)
        
    ax.imshow(g, aspect='auto', interpolation='none', vmin=0, vmax=1, extent=_shape)
    ax.xaxis.tick_top()

    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')

    dateaxis.set_major_formatter(mpl.dates.DateFormatter('%b %y'))
    sensoraxis.set_minor_locator(mpl.ticker.FixedLocator([x for x in range(n_sensors)]))

    sensoraxis.set_major_locator(mpl.ticker.FixedLocator([x+.5 for x in range(n_sensors)]))
    sensoraxis.set_major_formatter(mpl.ticker.FixedFormatter(labels))

    plt.tight_layout()
    return ax