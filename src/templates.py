import numpy as np
import pandas as pd
from pandas import IndexSlice as IDX
import matplotlib.pyplot as plt
import matplotlib as mpl
import src.style as style
from src.style import BUID, APPS, ROOMS, CLRS
import src.toolbox as tb
import src.preprocessing as pp
try:
    from src.preprocessing import BUID, ROOMS, APPS, DB, AMB, IND
except ImportError:
    pp.getDB()
    from src.preprocessing import BUID, ROOMS, APPS, DB, AMB, IND

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

def plotOverview(bui='LB', app='N', start='2022-01-01', end='2022-01-03', show_window_ticks = True):
    
    spacing = 2.2
    markersize = 4
    markevery = 120
    _lw = .5
    datum_von = pd.to_datetime(start).tz_localize('Europe/Berlin')
    datum_bis = pd.to_datetime(end).tz_localize('Europe/Berlin')

    _data = DB[bui][app].loc[datum_von:datum_bis]

    fig, axs = plt.subplots(6,1, **style.size(1.3), sharex=True)
    #fig.suptitle(f'{BUID[bui]} | Wohnung {APPS[app]}\n{datum_von:%d.%m.%Y} bis {datum_bis:%d.%m.%Y}', y=1.05)
    fig.suptitle(f'{BUID[bui]} | Wohnung {APPS[app]}', y=1.05)
    # Hauptlegende
    leg = {}
    for room in _data.columns.get_level_values(0).unique():
        try:
            p2 = mpl.lines.Line2D([0],[0],color=CLRS[room])
            p1 = mpl.patches.Patch(color = CLRS[room], linewidth=0, alpha=0.15)
            leg[ROOMS[room]] = (p1,p2)
        except KeyError:
            pass
    fig.legend([item for key, item in leg.items()], [key for key, item in leg.items()], ncol=4, bbox_to_anchor=(0.5,.985), frameon=True, loc='lower center')

    # Fensteröffnung
    ax = axs[2]

    i = spacing
    ticks = [0]
    ticklabels = ['']
    for window, data in _data[['K', 'WZ','SZ']].filter(regex='Fenster ').resample('15T').max().asfreq('T',method='ffill').iteritems():
        ax.fill_between(x=data.index, y1=data * ticks[-1], y2=data * i , alpha=0.5, label=window, zorder=0, color = CLRS[window[0]], edgecolor='None')
        ax.plot(data * i- spacing/2, linestyle = 'None',marker='_', zorder=1, markevery=1, color = CLRS[window[0]], label = f'{window[0]}')
        ticks.append(i)
        windowid = f'{window[1].split(" ", 1)[1]}'
        ticklabels.append(windowid)
        i += spacing
    ax.set_yticks([x - spacing/2 for x in ticks[1:]])
    ax.set_yticklabels(ticklabels[1:])
    ax.set_title('Fensteröffnung')

    # Lufttemperaturen
    ax = axs[0]
    ax1 = ax.twinx()
    ax1.set_ylim(0,1)
    ax1.yaxis.set_visible(False)
    _anw = DB.loc[datum_von:datum_bis, (bui, app, 'WE', 'Anwesenheit')]
    ax1.fill_between(_anw.index, _anw, color='k', alpha=.15)

    ## Raum
    ax.plot(_data['K'].filter(like='Tair'), label = r'Tair [Küche]', zorder=5, c=CLRS['K'],marker = 'x', lw = _lw, ms=markersize, mfc='None', markevery=markevery)
    ax.plot(_data['WZ'].filter(like='Tair'), label = r'Tair [Wohnzimmer]', zorder=5, c=CLRS['WZ'],marker = 'x', lw = _lw, ms=markersize, mfc='None', markevery=markevery)
    ax.plot(_data['SZ'].filter(like='Tair'), label = r'Tair [Schlafzimmer]', zorder=5, c=CLRS['SZ'],marker = 'x', lw = _lw, ms=markersize, mfc='None', markevery=markevery)
    ax.plot(_data['B'].filter(like='Tair'), label = r'Tair [Bad]', zorder=5, c=CLRS['B'],marker = 'x', lw = _lw, ms=markersize, mfc='None', markevery=markevery)


    ## Außenluft
    ax.plot(_data['SZ']['T_amb'], label = r'Tair [ambient]', zorder=50, color='k', marker = 'o', lw = _lw, ms=markersize, mfc='None', markevery=markevery)

    han = [
        mpl.lines.Line2D([0],[0],color='k', marker = 'x',mfc='None', ms = markersize, label = r'$T_{air}$'), 
        mpl.lines.Line2D([0],[0],color='k', marker = 'o',mfc='None', ms = markersize, label = r'$T_{amb}$'),
        #mpl.lines.Line2D([0],[0],color='k', marker = 'v',mfc='None', ms = markersize, label = 'Tset'),
        mpl.patches.Patch(color = 'k',label='Anwesenheit', linewidth=0, alpha=0.15)]
    ax.legend(handles=han, ncol=4, bbox_to_anchor=(1,1), frameon=False, loc='lower right')
    #ax.set_ylabel('$\si{\celsius}$')
    ax.set_title('Lufttemperaturen')
    ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(5))
    ax.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(2.5))
    ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:2n}°C'))

    # Heizkörper
    ## Oberflächentemperatur
    ax = axs[1]
    ax.set_title('Heizkörpertemperatur (Oberfläche)')
    ax.plot(_data['K'].filter(like='Thk'), label = 'HK [Küche]', zorder=5, c=CLRS['K'],marker = 'v', lw = _lw, ms=markersize, mfc='None', markevery=markevery)
    ax.plot(_data['WZ'].filter(like='Thk'), label = 'HK [Wohnzimmer]', zorder=5, c=CLRS['WZ'],marker = 'v', lw = _lw, ms=markersize, mfc='None', markevery=markevery)
    ax.plot(_data['SZ'].filter(like='Thk'), label = 'HK [Schlafzimmer]', zorder=5, c=CLRS['SZ'],marker = 'v', lw = _lw, ms=markersize, mfc='None', markevery=markevery)
    ax.plot(_data['B'].filter(like='Thk'), label = 'HK [Bad]', zorder=5, c=CLRS['B'],marker = 'v', lw = _lw, ms=markersize, mfc='None', markevery=markevery)
    ax.legend(
        handles = [mpl.lines.Line2D([0],[0],color='k', marker = 'v',mfc='None', ms = markersize, label = r'$T_{HK,sur}$')], 
        ncol=4, 
        bbox_to_anchor=(1,1), 
        frameon=False,
        loc='lower right')
    #ax.set_ylabel('$\si{\celsius}$')
    ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(10))
    ax.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(5))
    ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:2n}°C'))
    xlim_Thk = {'MH': 65, 'MW':75, 'LB': 55}
    ax.set_ylim(20,xlim_Thk[bui])

    ## Klassifikation
    ax2 = ax.twinx()
    ax2.fill_between(_data['SZ']['HK_IO'].index, y1 = 1, y2 = 1 + _data['SZ']['HK_IO'],color=CLRS['SZ'], alpha=0.25, ec='none')
    ax2.fill_between(_data['WZ']['HK_IO'].index, y1 = 2, y2 = 2 + _data['WZ']['HK_IO'],color=CLRS['WZ'], alpha=0.25, ec='none')
    ax2.fill_between(_data['B']['HK_IO'].index, y1 = _data['B']['HK_IO'], color=CLRS['B'], alpha=0.25, ec='none')
    #ax2.set_yticks([0.5, 1.5, 2.5], ['B', 'WZ', 'SZ'])
    ax2.set_ylim(0,3)
    #ax2.set_ylabel('Klassifikation (An / Aus)')
    ax2.yaxis.set_visible(False)

    # Raumluftfeuchte
    ax = axs[3]
    ax.plot(_data['K'].filter(like='Rh'), label = 'Küche', zorder=5, c=CLRS['K'],marker = 'v', lw = _lw, ms=markersize, mfc='None', markevery=markevery)
    ax.plot(_data['WZ'].filter(like='Rh'), label = 'Wohnzimmer', zorder=5, c=CLRS['WZ'],marker = 'v', lw = _lw, ms=markersize, mfc='None', markevery=markevery)
    ax.plot(_data['SZ'].filter(like='Rh'), label = 'Schlafzimmer', zorder=5, c=CLRS['SZ'],marker = 'v', lw = _lw, ms=markersize, mfc='None', markevery=markevery)
    ax.plot(_data['B'].filter(like='Rh'), label = 'Bad', zorder=5, c=CLRS['B'],marker = 'v', lw = _lw, ms=markersize, mfc='None', markevery=markevery)
    ax.plot(_data['SZ']['Rh_amb'], label = 'Ambient', zorder=50, color='k', marker = 'o', lw = _lw, ms=markersize, mfc='None', markevery=markevery)
    ax.set_title('relative Luftfeuchte')
    han = [
        mpl.lines.Line2D([0],[0],color='k', marker = 'v',mfc='None', lw = _lw, ms = markersize, label = r'$rH$'), 
        mpl.lines.Line2D([0],[0],color='k', marker = 'o',mfc='None', ms = markersize, lw = _lw, label = r'$rH_{AMB}$')
    ]
    ax.legend(handles=han, ncol=4, bbox_to_anchor=(1,1), frameon=False, loc='lower right')
    ax.set_ylim(20,100)
    ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(20))
    ax.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(10))
    ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(100))

    # CO2 - Konzentration
    ax = axs[4]
    ax.plot(_data['SZ'].filter(like='CO2'), label = 'Schlafzimmer', lw = _lw, zorder=5, c=CLRS['SZ'])
    ax.set_title('CO2 Konzentration')
    ax.set_ylim(0)
    ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(1000))
    ax.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(500))
    ax.set_ylabel('$[ppm]$', loc='top', rotation=0, labelpad=-15)
    # Wärmemengenzähler
    ax = axs[5]
    ax.set_title('Wärmemengenzähler')
    _wärmemenge = _data['WE']['Wärmemenge'].diff().resample('30T').mean().asfreq('T', 'ffill')
    ax.fill_between(_wärmemenge.index, _wärmemenge, alpha=0.5, color='Darkred', label='Wärmemenge', lw=0)
    ax.set_ylabel(r'$[\si{\kilo\watt\hour}]$', loc='top', rotation=0, labelpad=-15)

    ## HK-Klassifikation Klassifikation
    ax2 = ax.twinx()
    ax2.fill_between(_data.filter(like='HK_IO').index, y1 = _data.filter(like='HK_IO').max(axis=1), label = r'$HK_{IO}$', color='k', alpha=0.15, ec='none')
    ax2.yaxis.set_visible(False)

    han, lab = [], []
    for _ax in [ax, ax1, ax2]:
        _h, _l = _ax.get_legend_handles_labels()
        han.extend(_h)
        lab.extend(_l)
    ax.legend(handles=han, labels=lab, ncol=2, bbox_to_anchor=(1,1), frameon=False, loc='lower right')

    # Markiere Fensteröffnungen
    def Draw_Window_Openings(_ax):
        for (room, window), group in _data.filter(regex='Fensteröffnung').iteritems():
            m = _ax.get_ylim()
            data = group.dropna()
            _ax.vlines(data.index, ymin=m[0], ymax=m[1], color=CLRS[room], label = 'F. Öffnung', lw=1, ls='dashed' ,alpha=0.5)
            _ax.set_ylim(m)

    for ax in axs[:5]:
        if show_window_ticks:
            Draw_Window_Openings(ax)
        ax.grid(which='both')
        ax.xaxis.remove_overlapping_locs = False
        ax.tick_params(width=0, length=15, which='major', axis='x', rotation=0)
        ax.xaxis.set_major_locator(mpl.dates.HourLocator([0]))
        ax.xaxis.set_minor_locator(mpl.dates.HourLocator([0,4,8,12,16,20]))
        ax.xaxis.set_major_formatter(mpl.dates.DateFormatter('%d.%b'))
        ax.xaxis.set_minor_formatter(mpl.dates.DateFormatter('%H:%M'))
    
    fig.tight_layout(pad=0.1)

def plotHK_IO_simple(bui, app, start, ende):
    fig, axs = plt.subplots(
        2, 1, figsize=style.DIN['A5L'], sharex=True, sharey=True)
    fig.suptitle(f'{BUID[bui]} | {APPS[app]}')
    axs1 = []
    axs2 = []
    axs3 = []
    for r, room in enumerate(['WZ', 'SZ']):
        df = DB[bui][app][room]
        ax = axs[r]
        ax.set_ylim(0, 60)
        ax.set_ylabel('Temperatur $[°C]$')
        ax.set_title(f'{ROOMS[room]}')

        ax.plot(df.loc[start:ende, 'Thk'], c='k', linestyle='none',
                marker='.', label='Heizkörpertemperatur')
        ax.plot(df.loc[start:ende, 'Tair'].mask((df.loc[start:ende, 'Fenster'] > 0)),
                c='r', linestyle='none', marker='.', label='Lufttemperatur')

        _df1 = df.loc[start:ende, 'Thk'].rolling(
            'H').median().resample('2H').median().asfreq('T', 'ffill')
        _df2 = df.loc[start:ende, 'Tair'].rolling(
            'H').median().resample('2H').median().asfreq('T', 'ffill')
        _df3 = (_df1 - _df2)
        _df3 = _df3[_df3 > 0]

        ax.plot(_df3, c='r', linestyle='none', marker='_',
                #label=r'$\delta(T_{HK}, T_{Raum})$'
                )
        ax.axhline(2.5)

        ax1 = ax.twinx()
        ax1.spines['right'].set_visible(True)
        ax1.set_ylabel('Fensteröffnung [n]')
        axs1.append(ax1)
        ax1.plot(df.loc[start:ende, 'Fenster'], linestyle='none',
                    marker='.', label='n(Fenster offen)', c='g')
        ax1.yaxis.set_major_locator(mpl.ticker.MultipleLocator(1.00))
        ax1.yaxis.set_major_formatter(
            mpl.ticker.StrMethodFormatter("{x:.0f}"))

        ax2 = ax.twinx()
        ax2.spines['right'].set_position(('axes', 1.09))
        ax2.spines['right'].set_visible(True)
        ax2.set_ylabel('Tset [°C]')
        ax2.set_ylim(18, 26)
        axs2.append(ax2)
        ax2.plot(df.loc[start:ende, 'Tset'],
                    linestyle='none', marker='_', c='g', label='Tset')

        ax3 = ax.twinx()
        ax3.spines['right'].set_position(('axes', 1.18))
        ax3.spines['right'].set_visible(True)
        ax3.set_ylabel('Heizung [AN|AUS]')
        ax3.set_ylim(0, 1)
        ax3.yaxis.set_major_locator(mpl.ticker.FixedLocator([0, 1]))
        ax3.yaxis.set_major_formatter(
            mpl.ticker.FixedFormatter(['AUS', 'AN']))
        axs3.append(ax3)
        hkio = df.loc[start:ende, 'HK_IO'].dropna()
        ax3.fill_between(hkio.index, 1, where=hkio, ec='None', alpha=0.25,
                            transform=ax.get_xaxis_transform(), label='Heizung')
        han = []
        lab = []
        for a in [ax, ax1, ax2, ax3]:
            _han, _lab = a.get_legend_handles_labels()
            han.extend(_han)
            lab.extend(_lab)

    sync_yaxis(axs1)
    axs[0].legend(handles=han, labels=lab, ncol=4, bbox_to_anchor=(
        1, 1), loc='lower right', frameon=False)
    fig.tight_layout()

def plotCompareLüftungAPP(bui, app, focus_room, _df):
    opening_time = _df.iloc[1]['Datetime']['first']
    closing_time = _df.iloc[-2]['Datetime']['last']

    duration = closing_time - opening_time

    _lower = max(_df.iloc[0]['Datetime']['first'], opening_time - (3*duration))
    _upper = min(_df.iloc[-1]['Datetime']['last'], closing_time+3*duration)

    df = DB.loc[_lower: _upper, (bui, app)].swaplevel(axis=1).sort_index(axis=1).dropna(how='all').dropna(how='all', axis=1)
    before = _df.iloc[0]
    after = _df.iloc[-1]

    rooms = df.columns.get_level_values(1).unique().to_list()
    rooms.remove('WE')

    colors = {
        'Tair': mpl.cm.Reds(np.random.default_rng().integers(100,255)), 
        'Thk': mpl.cm.Reds(np.random.default_rng().integers(50,180)),
        'Fenster' : mpl.cm.Greens(np.random.default_rng().integers(100,255))
        }

    fig, axs = plt.subplots(len(rooms)+1, 1, sharex=True, figsize=style.DIN['A4L'])
    axsl = []
    order = {}
    for i, room in enumerate(rooms):
        order[room] = i
        axsl.append(axs[i].twinx())
        axsl[i].set_ylim(20,70)
        axs[i].set_title(room)
        axs[i].set_xlim(_lower,_upper)
        axs[i].plot(df['Tair'][room], color=colors['Tair'], marker='x')
        if room in df['Thk']:
            axsl[i].plot(df['Thk'][room], color=colors['Thk'], ms=5, markevery=10)
            axsl[i].fill_between(df.index, y1=axsl[i].get_ylim()[1], where=df['HK_IO'][room] > 0, color=colors['Thk'], alpha=0.1)
        try:
            for n in range(4):
                if room == focus_room:
                    _lw = 4
                    _color = 'gold'
                else:
                    _lw = 0
                    _color = colors['Fenster']
                axsl[i].fill_between(df.index, y1=axsl[i].get_ylim()[1], where=df['Fenster'][room] > n, color=_color, alpha=0.2, lw=_lw)
        except KeyError:
            pass
        
    axs[order[focus_room]].hlines(y=before['Tair']['median'], xmin=before['Datetime']['first'], xmax = opening_time, linewidth=2)

    axs[order[focus_room]].hlines(y=after['Tair']['median'], xmin=closing_time, xmax = after['Datetime']['last'], linewidth=2)

    axs[-1].set_title('Wetter')
    axs[-1].plot(df['T_amb'].max(axis=1))
    axs[-1].plot(df['T_amb_g24'].max(axis=1), linestyle='dashed')

    for i in range(len(axs)-2):
        axs[i].get_shared_y_axes().join(axs[i], axs[i+1])
        axsl[i].get_shared_y_axes().join(axsl[i], axsl[i+1])

    fig.suptitle(f'{BUID[bui]} {APPS[app]} {ROOMS[focus_room]}')
    
    fig.tight_layout()

def plotLüftung(bui, app, room):
    _id, df_airnode, df, data = pp.getLüftungsDatasets(bui, app, room)
    fig, ax = plt.subplots(sharex=True)
    fig.suptitle(f'{BUID[bui]} | Wohnung {APPS[app]} | {ROOMS[room]}')
    # Raumlufttemperatur
    ax.plot(df_airnode['Tair'])

    # Außenklima
    try:
        ax.plot(df_airnode['T_amb'])
        ax.plot(df_airnode['T_amb_g24'])
    except KeyError:
        print('Keine Wetterdaten vorhanden....')
        print('Datensatz wird trotzdem verwendet!')

    # Median in Periode vor der Lüftung (Gerade)
    ax.hlines(y = data['median_before'], xmin=df_airnode.index.min(), xmax=data['idx_opening'], lw=2, color='green')
    # Median in der Periode nach der Lüftung (Gerade)
    ax.hlines(y = data['median_after'], xmax=df_airnode.index.max(), xmin=data['idx_closing'], lw=2, color='green')

    # Markiere die Temperaturen (Tair, median(before), Tmin) zum Zeitpunkt der Fensteröffnung
    ax.plot([data['idx_opening'], data['idx_opening'], data['idx_opening']], [df_airnode.loc[data['idx_opening'], 'Tair'], data['median_before'], data['Tmin']], marker = 'x', ms = 5, ls='None')

    # Markiere die minimale Temperatur während der Lüftung
    ax.plot(data['idx_Tmin'], data['Tmin'], marker = 'x', ms = 5)
    #ax.vlines(_data['idx_Tmin'], ymin = ax.get_ylim()[0], ymax=_data['Tmin'])
    #ax.hlines(_data['Tmin'], xmin = ax.get_xlim()[0], xmax=_data['idx_Tmin'])

    ## Markiere die Dauer, bis das Temperaturminimum erreicht wird.
    ax.hlines(y = data['Tmin'], xmin = data['idx_opening'], xmax = data['idx_Tmin'], ls='dashed', color='red', lw=1)

    # Markiere den Punkt an dem der Median der nachfolgenden Periode erreicht wird.
    ax.plot(data['idx_median_after'], data['median_after'], marker = 'x', ms = 5)
    ax.vlines(data['idx_median_after'], ymin = data['Tmin'], ymax=data['median_after'], color='blue', lw=1)

    # Markiere Lufttemperatur zu Beginn der Lüftung.
    ax.vlines(data['idx_opening'], ymin = ax.get_ylim()[0], ymax=df_airnode.loc[data['idx_opening'], 'Tair'])

    # Markiere die Aufwärmzeiten nach der Lüftung.

    ## Fenster geöffnet
    ax.hlines(y = data['Tmin'], xmin = data['idx_Tmin'], xmax = data['idx_closing'], ls='dotted', color='blue', lw=1)
    ## Fenster geschlossen
    ax.hlines(y = data['Tmin'], xmin = data['idx_closing'], xmax = data['idx_median_after'], ls='dashed', color='blue', lw=1)

    # Plotte stabilisierte Temperatur, wenn diese ungleich des Medians ist.                    
    if isinstance(data['idx_stable'], pd.Timestamp):
        if round(data['Tstable']) < round(data['median_after']):
            ax.axhline(data['Tstable'], color='k', ls='dashed')

    # Plotte Fensteröffnung
    ## Nachbarräumde
    ax1 = ax.twinx()
    for i in range(int(df_airnode['Fenster_gesamt'].max())):
        ax1.fill_between(df_airnode.index, ax1.get_ylim()[1], where = df_airnode['Fenster_gesamt'] > i, color = 'brown', alpha=0.2)
    ax1.plot(df_airnode.index, df_airnode['Fenster_gesamt'], color='brown')

    ## Focus-Airnode
    for i in range(int(df_airnode['Fenster'].max())):
        ax1.fill_between(df_airnode.index, ax1.get_ylim()[1], where = df_airnode['Fenster'] > i, color = 'darkgreen', alpha=0.2)

    # Plotte Heizungsaktivierung
    try:
        for i in range(int(df_airnode['HK_IO'].max())):
            ax1.fill_between(df_airnode.index, ax1.get_ylim()[1], where = df_airnode['HK_IO'] > i, color = 'skyblue', alpha=0.2)
    except KeyError:
        pass

    # Graph-Layout
    ax.set_xlim(df_airnode.index.min(), df_airnode.index.max())
    fig.tight_layout(pad=0.1)

def plotLüftungsKlassifikation():
    DB_WINDOW = DB.loc[(DB.loc[:, IDX[:, :, :, 'Fensteröffnung']] > 0).any(axis=1) | (
        DB.loc[:, IDX[:, :, :, 'Fensteröffnung']] < 0).any(axis=1), IDX[:, :, ['WZ', 'SZ', 'K'], :]]

    for (bui, app), df in DB_WINDOW.groupby(level=[0, 1], axis=1):
        df = df.droplevel([0, 1], axis=1).dropna(how='all')
        fig, ax = plt.subplots(figsize=(15, 5))
        fig.suptitle(
            f'{BUID[bui]} | Wohnung {APPS[app]}\nZeitraum von {df.index.min():%d.%m.%y} bis {df.index.max():%d.%m.%y}')

        # Temperatur, Links
        ax.set_ylim(10, 30)
        ax.set_zorder(1)
        ax.patch.set_visible(False)
        ax.set_ylabel('Temperatur [°C]')

        # Fensteröffnung, fill_between, rechts, 1
        ax1 = ax.twinx()
        ax1.set_zorder(0)
        ax1.spines["right"].set_visible(True)
        ax1.patch.set_visible(False)
        ax1.set_ylabel('Anzahl geöffneter Fenster')

        # Häufigkeit Heizung An + Fenster Auf, rechts, 2
        ax3 = ax.twinx()
        ax3.spines["right"].set_position(("axes", 1.05))
        ax3.set_zorder(2)
        ax3.spines["right"].set_visible(True)
        ax3.patch.set_visible(False)
        ax3.set_ylabel('Fenster offen + Heizung an')
        ax3.set_ylim(0, 1.1)
        ax3.set_clip_on(False)
        ax3.yaxis.set_major_locator(mpl.ticker.FixedLocator([0, 1]))
        ax3.yaxis.set_major_formatter(mpl.ticker.FixedFormatter(['aus', 'an']))

        leg1 = []
        for c, (room, data) in enumerate(df.groupby(level=0, axis=1)):
            data = data.droplevel(level=0, axis=1)
            try:
                rtemp = ax.plot(data[(data['Fensteröffnung'] > 0)]['Tair'], c=plt.cm.Set2(
                    c), label=room, linestyle='None', marker='_', mfc='none', ms=5)
            except KeyError as error:
                #print(f'{room}: Kann Raumtemperaturen nicht plotten, {error} wurde nicht gefunden.')
                pass

            try:
                _df = data['HK_IO'][data['Fenster'] > 0]
                sc = ax3.scatter(x=_df.where(_df > 0).dropna().index, y=_df.where(
                    _df > 0).dropna(), s=25, color=plt.cm.Set2(c), marker='o', facecolor='None')
            except KeyError as error:
                #print(f'{room}: Kann Heizkörpertemperatur nicht plotten, {error} wurde nicht gefunden.')
                pass

            try:
                _dfp = data['Fensteröffnung'].cumsum().asfreq('T', 'ffill')
                ax1.fill_between(_dfp.index, _dfp, color=plt.cm.Set2(
                    c), label=room, alpha=0.25)
                ax1.plot(_dfp.rolling('7d').mean(), color=plt.cm.Set2(c),
                        linestyle='dashed', linewidth=1, clip_on=False, )
            except KeyError as error:
                print(
                    f'{room}: Kann Fensteröffnung nicht plotten, {error} wurde nicht gefunden.')
                pass

            p1 = mpl.patches.Patch(color=plt.cm.Set2(
                c), linewidth=0, alpha=0.25, label=ROOMS[room])
            leg1.append(p1)

        # Legenden
        p3 = mpl.lines.Line2D([0], [0], color='k', linestyle='None',
                            marker='_', markersize=5, label='Raumlufttemperatur')
        p4 = mpl.lines.Line2D([0], [0], color='k', linestyle='None',
                            marker='o', mfc='None', markersize=5, label='Heizung An')

        leg1 = ax.legend(handles=leg1, ncol=3, bbox_to_anchor=(
            0, 1), loc='lower left', frameon=False)
        leg2 = ax.legend(handles=[p3, p4], ncol=3, bbox_to_anchor=(
            0.5, 1), loc='lower center', frameon=False)
        #leg3 = ax1.legend(handles=leg3, ncol=5, bbox_to_anchor=(1,1), loc='lower right', frameon=False)
        ax.add_artist(leg1)

        # y-Achse
        # ax1.set_yticks(range(int(ax1.get_yticks().max().round(0))))
        # ax2.set_yticks(range(int(ax2.get_yticks().max().round(0))+1))
        ax1.yaxis.set_major_locator(mpl.ticker.FixedLocator(
            np.arange(max(ax1.get_ylim())+1, step=1)))
        ax1.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.0f'))

        # x-Achse
        ax.set_xlim(df.index.min(), df.index.max())

        if (df.index.max() - df.index.min()).days <= 2:
            ax.xaxis.set_minor_locator(mpl.dates.HourLocator(interval=4))
            ax.xaxis.set_minor_formatter(mpl.dates.DateFormatter('%H:%M'))

        elif 2 < (df.index.max() - df.index.min()).days <= 7:
            ax.xaxis.set_minor_locator(mpl.dates.HourLocator(12))
            ax.xaxis.set_minor_formatter(mpl.dates.DateFormatter('%H:%M'))

        else:
            ax.xaxis.set_major_locator(mpl.dates.DayLocator(1))
            ax.xaxis.set_minor_locator(mpl.dates.DayLocator(16))
            ax.xaxis.set_minor_formatter(mpl.dates.DateFormatter('%b %y'))
            ax.xaxis.set_major_formatter(mpl.dates.DateFormatter(''))

        fig.tight_layout()
