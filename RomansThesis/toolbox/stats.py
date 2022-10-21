import warnings
import statsmodels as sm
import scipy.stats as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from .. import style
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.feature_selection import SequentialFeatureSelector, SelectFromModel
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import auc, RocCurveDisplay
from sklearn.model_selection import StratifiedKFold
from scipy.stats import ks_2samp
from time import time
from scipy import stats

# Create models from data
def best_fit_distribution(data, bins=200, ax=None):
    """Model data by finding best fit distribution to data
    
    returns:
    
    dist, params, sse"""
    from timeit import default_timer as timer
    from datetime import timedelta

    
    # Get histogram of original data
    y, x = np.histogram(data, bins=bins, density=True)
    x = (x + np.roll(x, -1))[:-1] / 2.0

    # Best holders
    best_distributions = []

    # Estimate distribution parameters from data
    distnames = st._continuous_distns._distn_names
    for ii, _distribution in enumerate([d for d in distnames if not d in ['levy_stable', 'studentized_range', 'ncf']]):
        start = timer()
        print(f"{ii+1:>3} / {len(distnames):<3}: {_distribution}", end='\r')

        distribution = getattr(st, _distribution)

        # Try to fit the distribution
        try:
            # Ignore warnings from data that can't be fit
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore')

                # fit dist to data
                params = distribution.fit(data)

                # Separate parts of parameters
                arg = params[:-2]
                loc = params[-2]
                scale = params[-1]

                # Calculate fitted PDF and error with fit in distribution
                pdf = distribution.pdf(x, loc=loc, scale=scale, *arg)
                sse = np.sum(np.power(y - pdf, 2.0))

                # if axis pass in add to plot
                try:
                    if ax:
                        pd.Series(pdf, x).plot(ax=ax)
                except Exception:
                    pass
                # identify if this distribution is better
                best_distributions.append((distribution, params, sse))
        except Exception as e:
            print(e)
            pass
        end = timer()
        print(f"{ii+1:>3} / {len(distnames):<3}: {_distribution} | {timedelta(seconds=end-start)}", end='\r')

    return sorted(best_distributions, key=lambda x: x[2])


def make_pdf(dist, params, size=1000):
    """Generate distributions's Probability Distribution Function """

    # Separate parts of parameters
    arg = params[:-2]
    loc = params[-2]
    scale = params[-1]

    # Get sane start and end points of distribution
    start = dist.ppf(0.01, *arg, loc=loc,
                     scale=scale) if arg else dist.ppf(0.01, loc=loc, scale=scale)
    end = dist.ppf(0.99, *arg, loc=loc,
                   scale=scale) if arg else dist.ppf(0.99, loc=loc, scale=scale)

    # Build PDF and turn into pandas Series
    x = np.linspace(start, end, size)
    y = dist.pdf(x, loc=loc, scale=scale, *arg)
    pdf = pd.Series(y, x)

    return pdf

def fit_dist(data, distname, _bins=None, ax=None, **kwargs):
    """
    returns:
        dist, params([args], loc, scale), sse
    """
    if _bins is None:
        y, x = np.histogram(data, density=True)
    else:
        y, x = np.histogram(data, bins=_bins, density=True)

    x = (x + np.roll(x, -1))[:-1] / 2.0

    try:
        dist = getattr(st, distname)
    except Exception as e:
        print(e)
        return

    # fit dist to data
    params = dist.fit(data)

    # Separate parts of parameters
    arg = params[:-2]
    loc = params[-2]
    scale = params[-1]

    # Calculate fitted PDF and error with fit in distribution
    pdf = dist.pdf(x, loc=loc, scale=scale, *arg)
    sse = np.sum(np.power(y - pdf, 2.0))

    if ax is None:
        plt.plot(make_pdf(dist, params), **kwargs)
    else:
        ax.plot(make_pdf(dist, params), **kwargs)
    return dist, params, sse

def ci(a, which=95, axis=None):
    """Return a percentile range from an array of values."""
    p = 50 - which / 2, 50 + which / 2
    return np.nanpercentile(a, p, axis)

def CorreleationMatrix(df,highlights:list=[], triangular=False,ax=None, **heatmap_kwargs):
    corr = df.corr(method='pearson')
    if triangular:
        heatmap_kwargs['mask'] = np.triu(np.ones_like(corr, dtype=bool))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    if ax is None:
        fig, ax = plt.subplots(**style.size(1))
    else:
        fig = ax.get_figure()

    kws = {'ax':ax, 
    'vmin':-1, 
    'vmax':1, 
    'annot':True, 
    'fmt':'.1f',
    'cmap': cmap, 
    'square':True, 
    'linewidths':.5, 
    'annot_kws':dict(size=8), 
    'cbar_kws': dict(use_gridspec=True,location="top",pad=0.075, shrink=.6, aspect=30, label='Pearson $R$')}

    kws.update(heatmap_kwargs)
    if len(highlights) > 0:
        corr = corr[highlights].T
    ax = sns.heatmap(corr, **kws)
    cbar = ax.collections[0].colorbar
    #cbar.ax.xaxis.tick_top() # x axis on top
    cbar.ax.xaxis.set_ticks_position('bottom')
    #plt.xticks(rotation=90)
    # labels = [tick.get_text() for tick in ax.get_xticklabels()]
    # N = len(labels)
    # for wanted_label in highlights:
    #     wanted_index = labels.index(wanted_label)
    #     x, y, w, h = 0, wanted_index, N, 1
    #     for _ in range(2):
    #         ax.add_patch(mpl.patches.Rectangle((x, y), w, h, fill=False, linestyle='dashed', edgecolor='crimson', lw=2, clip_on=False))
    #         x, y = y, x # exchange the roles of x and y
    #         w, h = h, w # exchange the roles of w and h
    ax.tick_params(length=0)
    fig.tight_layout()
    return ax

# Variance Inflation Factor for Multicollinarity
def VIFMatrix(df, triangular=True, ax=None, annot=True, tresh=10, **heatmap_kwargs):
    f"""Variance inflation Factor:

        | VIF value | Diagnosis                                        |
        | --------- | ------------------------------------------------ |
        | <1        | Complete absence of multicollinearity            |
        | 1 - tresh | Absence of strong multicollinearity              |
        | > tresh   | Presence of moderate to strong multicollinearity |

        treshold: float, default = 5
        """
    if ax is None:
        fig, ax = plt.subplots(**style.size(1))
    else:
        fig = ax.get_figure()
    bounds = [0, 1, tresh, 100]
    norm = mpl.colors.BoundaryNorm(boundaries=bounds, ncolors=256)
    cbarticks = bounds[:-1]
    cbarticklabels = bounds[:-1]
    cbarticklabels[-1] = f'> {cbarticklabels[-1]}'
    lab = ['keine\nMultikollinearität', 'keine starke\nMultikollinearität', 'starke\nMultikollinearität']
    minorlocs = [np.mean(item) for item in [*zip(bounds[1:], bounds[:-1])]]
    corr = (1 / (1 - df.corr()))
    heatmap_kws = dict(
        norm=norm, 
        cmap=style.Reds,
        linewidths=1, 
        ax=ax, 
        square=True, 
        cbar=True, 
        cbar_kws=dict(use_gridspec=True,location="top", pad=0.1, shrink=.6, aspect=30, label='Variance Inflation Factor $VIF$'),
        )
    if annot:
        heatmap_kws['annot'] = True
        heatmap_kws['annot_kws'] = dict(size=8)
        heatmap_kws['fmt'] = '.2n'
    if triangular:
        heatmap_kws['mask'] = np.triu(np.ones_like(corr, dtype=bool))
    heatmap_kws.update(heatmap_kwargs)
    ax = sns.heatmap(data=corr, **heatmap_kws)
    cbar = ax.collections[0].colorbar
    cbar.ax.set_xticks(minorlocs, lab, minor=True)
    cbar.ax.set_xticks(cbarticks, cbarticklabels)
    cbar.ax.tick_params(axis='x', which='minor', bottom=False, top=False, labeltop=False, labelbottom=True)
    cbar.ax.tick_params(axis='x', which='major', bottom=True, top=True)
    ax.set(xlabel=None, ylabel=None)
    fig.tight_layout()
    return ax

def qqplot(obs, exp, ax=None, test='None', bins=20, ci=0.95, n = 100, plot_kws=dict(), line_kws=dict(), **kwargs):

    obs_bin, bins = np.histogram(obs, bins=bins, density=False)
    exp_bin, bins = np.histogram(exp, bins=bins, density=False)
    exp_normbin = ((exp_bin/sum(exp_bin)) * sum(obs_bin))

    a = (1-ci)/2
    percs = np.linspace(a*100, 100-a*100,n)
    qn_a = np.percentile(obs, percs)
    qn_b = np.percentile(exp, percs)

    if ax is None:
        fig, ax = plt.subplots(**style.size(.3))
    else:
        fig = ax.get_figure()
    kwargs = dict(ls="", marker="o", markerfacecolor=mpl.colors.to_rgba(style.clrs[0], alpha=.5), markeredgecolor=mpl.colors.to_rgba(style.clrs[0], alpha=1))
    kwargs.update(plot_kws)
    ax.plot(qn_a, qn_b, **kwargs)

    if test == 'pearson':
        r, p_r = st.pearsonr(obs_bin, exp_normbin)
        test = f'Pearson Test:\nR: {r:.2n}\n $p_0$: {round(p_r*100, 2):.2n} \si{{\percent}}'
        ax.text(0.05,0.95,test, va='top', ha='left', transform=ax.transAxes, bbox=style.bbox_props)
        
    elif test == 'ks':
        s, p = ks_2samp(obs, exp)
        test = f'Kolmogorov-Smirnov Test:\ns: {s:.2n}\n $p_0$: {round(p*100, 2):.2n} \si{{\percent}}'
        ax.text(0.05,0.95,test, va='top', ha='left', transform=ax.transAxes, bbox=style.bbox_props)
    
    x = np.linspace(np.min((qn_a.min(),qn_b.min())), np.max((qn_a.max(),qn_b.max())))
    kwargs = dict(color="k", ls="--")
    kwargs.update(line_kws)
    ax.plot(x,x,  **kwargs)
    ax.set(xlabel='beobachtete Quantile [-]', ylabel='erwartete Quantile [-]')
    fig = plt.gcf()
    fig.tight_layout()
    return ax

def histanalyse(obs, exp, obs2 = None, ax=None, kind='hist', xlabel=None, bins=20, bw=1):
    obs_bin, bins = np.histogram(obs, bins, density=False)
    if obs2 is not None:
        obs2_bin, bins = np.histogram(obs2, bins, density=False)
    exp_bin, bins = np.histogram(exp, bins = bins, density=False)
    
    if ax is None:
        fig,ax = plt.subplots(**style.size(0.3))
    else:
        fig = ax.get_figure()
    if kind.lower() == 'kde':
        ax = sns.kdeplot(data=obs, bw_adjust=bw, label='Fensteröffnungen', ax=ax, color=style.clrs[0], fill=True)
        if obs2 is not None:
            ax = sns.kdeplot(data=obs2, bw_adjust=bw, label='Fensterschließungen', ax=ax, color=style.clrs[2], fill=True)
        ax = sns.kdeplot(data=exp, bw_adjust=bw, label='Referenz', ax=ax, fill=True, color=style.clrs[1])
    elif kind.lower() == 'hist':
        ax = sns.histplot(data=obs, stat='density', bins = bins, color=style.clrs[0], label='Fensteröffnungen', ax=ax)
        if obs2 is not None:
            ax = sns.histplot(data=obs2, stat='density', bins = bins, color=style.clrs[2], label='Fensterschließungen', ax=ax)
        ax = sns.histplot(data=exp, stat='density',bins = bins, color=style.clrs[1], label = 'Referenz', ax=ax)
    else:
        raise AttributeError('kind must be "hist" or "kde"')
    ax.legend()
    ax.set(xlabel=xlabel, ylabel='rel. Häufigkeit')
    fig.tight_layout()
    return ax


def ROCplot(X,y, scaler=StandardScaler(), classifier=LogisticRegression(solver='liblinear', class_weight='balanced'), n_splits=10, ax=None):
    features = list(X.columns)
    scaler.fit(X,y)
    n_samples, n_features = X.shape

    # Run classifier with cross-validation and plot ROC curves
    cv = StratifiedKFold(n_splits=n_splits)
    
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    if ax is None:
        fig, ax = plt.subplots(**style.size(.4))
    else:
        fig = ax.get_figure()

    for i, (train, test) in enumerate(cv.split(X, y)):
        classifier.fit(scaler.transform(X.iloc[train]), y.iloc[train])
        viz = RocCurveDisplay.from_estimator(
            classifier,
            scaler.transform(X.iloc[test]),
            y.iloc[test],
            name=f"$ROC_{i}$",
            alpha=0.3,
            lw=1,
            ax=ax,
        )
        interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(viz.roc_auc)

    ax.plot([0, 1], [0, 1], linestyle="--", lw=2, color=style.red, alpha=0.8)
    ax.legend().remove()
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    p1, = ax.plot(
        mean_fpr,
        mean_tpr,
        color=style.blue,
        label=f"AUC_{{mean}} = {mean_auc:.2n} $\\pm$ {std_auc:.2n})",
        alpha=0.8,
    )
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)

    ax.fill_between(
        mean_fpr,
        tprs_lower,
        tprs_upper,
        color="grey",
        alpha=0.2,
        label=r"$\pm$ 1 std. dev.",
    )

    p2 = mpl.patches.Patch(color='grey', alpha=0.2)
    
    feature_str = ', '.join(X.columns)
    if len(feature_str) > 90:
        feature_str = '\n'.join([', '.join(line) for line in np.split(np.array(X.columns), 2)])
    ax.set(
        xlim=[-0.05, 1.05],
        ylim=[-0.05, 1.05],
        title=f"Prädikatoren:\n{feature_str}",
        ylabel='Anteil korrekt Positiv',
        xlabel='Anteil falsch Positiv'
    )
    fig.legend(loc="upper left", bbox_to_anchor=(0, 0), ncol=3)

    fig.tight_layout()
    return classifier, scaler

def ecdf(a):
    x, counts = np.unique(a, return_counts=True)
    cusum = np.cumsum(counts)
    return x, cusum / cusum[-1]

def plot_ecdf(a):
    x, y = ecdf(a)
    x = np.insert(x, 0, x[0])
    y = np.insert(y, 0, 0.)
    plt.plot(x, y, drawstyle='steps-post')
    plt.grid(True)