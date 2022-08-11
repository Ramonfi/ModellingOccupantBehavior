import datetime as dt
import os
from tkinter import N
from zoneinfo import ZoneInfo
tz = ZoneInfo('Europe/Berlin')

import os, re, shutil, itertools, errno, csv

import scipy.stats as st
from scipy.stats import norm, lognorm, uniform, qmc

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from RomansThesis.toolbox.utils import running_bar
from RomansThesis.toolbox.comf import KelvinstundenEN, KelvinstundenNA
tab = ' '

class Weekschedule:
    def __init__(self, **kwargs):
        self.Name = 'Template'
        self.Days = '1 2 3 4 5 6 7'
        for key, item in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, item)

    def setWeek(self, weekarray):
        self.Name = self.Name.upper()
        self.Hourly = ''
        self.Daysarray = []
        self.Hoursarray = []
        self.Valuesarray = []
        Meansarray = []
        if isinstance(weekarray, list) and len(weekarray) == 7:
            for item in weekarray:
                self.Hourly = self.Hourly + item.Name + tab
                self.Daysarray.append(item.Name)
                self.Hoursarray.append(item.Hours)
                self.Valuesarray.append(item.Values)
                Meansarray.append(item.Mean)
            self.Mean = sum(Meansarray)/7
            return self
        elif isinstance(weekarray, Schedule):
            for i in range(1,8):
                self.Hourly = self.Hourly + weekarray.Name + tab
                self.Daysarray.append(weekarray.Name)
                self.Hoursarray.append(weekarray.Hours)
                self.Valuesarray.append(weekarray.Values)
                Meansarray.append(weekarray.Mean)
            self.Mean = sum(Meansarray)/7
            return self         
        else:
            raise ValueError('weekarray muss alle 7 Wochentage enthalten!')
    
    def __repr__(self):
        return 'WEEKLY SCHEDULE INSTANCE <{}>: Mittelwert {:.2f}'.format(self.Name, self.Mean)


class Schedule:
    def __init__(self, **kwargs):
        self.Name = 'Template'
        self.Type = 'daily'
        self.Timestep = .25
        self.Hours = []
        self.Values = []
        self.Mean = 0
        for key, item in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, item)
        self.dict = {i: 0 for i in self.timerange(0,24,self.Timestep)}
        self.Data = []

    def timerange(self, start, stop=None, step=None):
        if step == None:
            step = 1.0
        start = start
        if stop == None:
            stop = start + 0.0
            start = 0.0
        else:
            stop = stop + step
        count = 0
        while True:
            temp = float(start + count * step)
            if temp >= stop:
                break
            yield temp
            count += 1
    def setHourArray(self, Values):
        if len(Values)/24 == 1/self.Timestep:
            for i, item in enumerate(Values):
                self.dict[list(self.dict.keys())[i]] = item 
        self.up()
    def setHourly(self, Value, Hour):
        Value = float(Value) if Value<=1 else float(1)
        if isinstance(Hour, str) and ':' in Hour:
            h, m = Hour.split(':')
            Hour = float(h) + float(m)/60
        else: Hour = float(Hour)
        if 0 <= Hour <= 24:
            for key, item in self.dict.items():
                if key == Hour:
                    self.dict[key] = Value
        self.up()
        print(self.dict)
    def setDomain(self, From, To , Value):
        s = round(float(From) * (1/self.Timestep)) / (1/self.Timestep)
        srb = round(float(To) * (1/self.Timestep)) / (1/self.Timestep)
        if 0 <= s < 24 and 0 <= srb <= 24:
            if s < srb:
                for i in self.timerange(s, srb-self.Timestep, self.Timestep):
                    self.dict[i] = Value
            elif srb < s:
                for i in self.timerange(0, srb-self.Timestep, self.Timestep):
                    self.dict[i] = Value
                for i in self.timerange(s, 24, self.Timestep):
                    self.dict[i] = Value
            self.up()
            return self.Data
        else:
            if 0 <= s < 24: raise ValueError('Startwert muss zwischen 0:00h und 24:00h liegen')
            if 0 <= srb <= 24: raise ValueError('Startwert muss zwischen 0:00h und 24:00h liegen')
    def scale(self, factor):
        if 0 < factor <= 1:
            for key, item in self.dict.items():
                self.dict[key] = item * factor
            self.up()
        else:
            raise ValueError('factor muss zwischen 0 und 1 liegen')
    def up(self):
        self.Name = self.Name.upper()
        self.Hours = []
        self.Values = []
        val = self.dict[0]
        self.Data = ['{:.2f} {:.2f}'.format(float(0), float(val))]
        for key, item in self.dict.items():
            if item == val:
                pass
            if item != val:
                val = item
                self.Data.append('{:.2f} {:.2f}'.format(float(key-1), float(val)))
                self.Hours.append('{:.2f}'.format(float(key-1)))
                self.Values.append('{:.2f}'.format(float(val)))
        end = list(self.dict.keys())[-1]
        end_value = self.dict[end]
        self.Data.append('{:.2f} {:.2f}'.format(float(end), float(end_value)))

        self.Hours = ' '.join(self.Hours)
        self.Values = ' '.join(self.Values)

        self.Mean = (sum(self.dict.values())-self.dict[0])/(len(self.dict.keys())-1)
    def __repr__(self):
        return 'DAILY SCHEDULE INSTANCE <{}>: Mittelwert {:.2f}'.format(self.Name, float(self.Mean))

def timerange(start, stop=None, step=None):
    if step == None:
        step = 1.0
    start = start
    if stop == None:
        stop = start + 0.0
        start = 0.0
    else:
        stop = stop + step
    count = 0
    while True:
        temp = float(start + count * step)
        if temp >= stop:
            break
        yield temp
        count += 1

class ScheduleCollection:
    def __init__(self,mode = 'WEEK'):
        self.Type = mode
        self.Names = []
        self.Schedules = []
        self.df = pd.DataFrame()

    def add(self,sched):
        if self.Type == 'DAY':
            self.Names.append(sched.Name)
            self.Schedules.append(sched)
            self.df = pd.DataFrame([sched.Data for sched in self.Schedules], index=self.Names)
        elif self.Type == 'WEEK':
            self.Names.append(sched.Name)
            self.Schedules.append(sched)
            self.df = pd.DataFrame([sched.Daysarray for sched in self.Schedules], index=self.Names)
        else: 
            print('mode is not correct...')
    
    def export(self, dest='./sim/robust-inputs'):
        fn = os.path.join(dest,f'{self.Type}SCHED_COLLECTION.csv')
        self.df.to_csv(fn)
        print(f'Erfolgreich unter {fn} gespeichert!')
    def __repr__(self):
        return f'{self.Type}SCHEDULE-COLLECTION: <{len(self.Names)} Schedules>'
    def __call__(self):
        return self.df

class LAYER:
    def __init__(self, **kwargs):

        """Args:
        Name: material layer name
        Conductivity: define conductivity [kJ/(hr.m.K)]
        Capacity: define capacity [kJ/(kg.K)]
        Density: define density [kg/m³]
        PERT: define the renewable primary energy resources [MJ/kg]
        PENRT: define the non-renewable primary energy resources [MJ/kg]"""

        self.Name = 'NoName'
        self.Pspacing = None
        self.Pdiameter = None
        self.Conductivity = None
        self.Capacity = None
        self.Density = None
        self.Pwallthickness = 0 
        self.Pconductivity = 0
        self.Cpfluid = 4.19  #KJ/(kg.K)
        self.Resistance = None
        self.Type = 'CL'
        self.PERT = 0
        self.PENRT = 0
        self.GWP = 0
        self.Usagetype = None   # 'FLOOR' or 'CEILING'
        self.Cc_pspacing = None
        self.Cc_pidiameter = None
        self.Cc_cpfluid = 4.19
        self.Sp_normpower = 318 #in kJ/(hr.m2)
        self.Sp_normmflow = 43 #in kg/(hr.m2)
        self.Normarea = 15 #m2
        self.Normnloop = 1
        self.Ucomb = "GAP"
        self.Uwrx = "F(DTSURFNORM)"
        self.Dtsurfnorm = 2  #K
        self.Mflow = 0
        self.Plength = None
        self.obdUUID = ''
        self.GWP = 0
        self.PERT = 0
        self.PENRT = 0
        self.UsedInWalls = []
        for key, arg in kwargs.items():
            try:
                getattr(self,key)
                setattr(self,key,arg)
            except AttributeError:
                print('Keyword Argument konnte nicht zugeordnet werden.')

    def createActiveLayer(self,UsageType, PipeSpacing, PipeDiameter,PipeLength, ActiveArea, ModuleArea, Coefficient = 10.497, Exponent = 1.085, SpecNormFlow=43, SpecMassFlow=23):
        '''
        Verlegeabstand von 15 cm = 5,8 m Rohr / m2
        Verlegeabstand von 12,5 cm = 6,8 m Rohr / m²
        Verlegeabstand von 10 cm = 8,8 m Rohr / m² 

        Args:
        ---
        UsageType: Für Fußbodenheizung 'FLOOR' für Bauteilaktivierung 'CEILING'
        PipeSpacing: Verlegeabstand zwischen den Leitungen in [m]
        PipeDiameter: Durchmesser der Leitungen in [m]
        PipeLength: Leitungslänge in [m]
        Coefficient: Koeffizient zur Berechnung der Nennleistung
        Exponent: Exponent zur Berechnung der Nennleistung

        '''
        self.Usagetype = UsageType
        self.PipeSpacing = PipeSpacing
        self.Pdiameter = PipeDiameter
        self.Plength = PipeLength
        self.Sp_normpower = Coefficient * 10**Exponent * ActiveArea / ModuleArea * 3.6  #in kJ/(hr.m?)
        self.Sp_normmflow = SpecNormFlow * ActiveArea / ModuleArea #in kg/(hr.m?)
        self.Normarea = ModuleArea
        self.Mflow = SpecMassFlow

    def createFromDB(self, Name, row = None, **kwargs):
        '''
        '''
        self.Name = Name.upper()
        if isinstance(row, pd.Series):
            for id in row.index:
                if hasattr(self, id) and not np.isnan(row[id]):
                    setattr(self, id, row[id]) 
        else:
            for key, arg in kwargs.items():
                if hasattr(self,key):
                    setattr(self,key,arg)
                else:
                    print('Keyword Argument konnte nicht zugeordnet werden.')

        if not self.Resistance and self.Conductivity:
            self.Mode = 'MassLayer'
        elif not self.Conductivity and self.Resistance:
            self.Mode = 'ResistanceLayer'
        self.check()
        return self

    def Use(self, WallID):
        '''
        '''
        if WallID not in self.UsedInWalls:
            self.UsedInWalls.append(WallID)

    def check(self):
        '''
        '''
        LCAerrors = []
        if self.Mode == 'MassLayer':
            if not self.Density or self.Density <= 0:
                print('Achtung! Dichte nicht korrekt definiert') 
            for key in ['GWP', 'PERT', 'PENRT']:
                if getattr(self, key) == 0:
                    LCAerrors.append(key)
        if self.Mode == 'ResistanceLayer':
            if self.Density or self.Density != 0:
                print('Achtung! Resistance Layer haben per Definition keine Dichte') 

        if len(LCAerrors) > 0:
            print(f'{self.Name}: {LCAerrors} nicht hinterlegt')



    def __call__(self):
        return pd.DataFrame(data=[[self.Name, self.Conductivity, self.Resistance, self.Capacity, self.Density, self.GWP, self.PERT, self.PENRT]],columns=['Name', 'Conductivity', 'Resistance', 'Capacity', 'Density', 'GWP', 'PERT', 'PENRT' ])
    
    def __repr__(self):
        if len(self.UsedInWalls) > 0:
            return f'<{self.Mode}> OBJECT <{self.Name}>: Cond. {self.Conductivity}, Cap. {self.Capacity}, Density. {self.Density} | Used in [{", ".join(self.UsedInWalls)}]'
        else:
            return f'<{self.Mode}> OBJECT <{self.Name}>: Cond. {self.Conductivity}, Cap. {self.Capacity}, Density. {self.Density} | Not yet Used'


class WALL:
    def __init__(self, **kwargs):
        """Name: construction name
        Layer: define the construction layers starting from interior to exterior
        Abs_Front: define solar absorptance at front side [%/100]
        Abs_Back: define solar absorptance at back side [%/100]
        H_Front: define convective heat exchange coefficient at front side [kJ/(h.m2.K)]
        H_Back: define convective heat exchange coefficient at back side [kJ/(h.m2.K)]"""
        self.Name = ''
        self.Layers = []
        self.Thicknesses = []
        self.Thickness = 0
        self.Abs_Back = 0.5
        self.Abs_Front = 0.5
        self.H_Back = 64.0
        self.H_Front = 11.0
        self.Mass = 0
        self.Uvalue = 0
        self.Materials = {}
        for key, arg in kwargs.items():
            try:
                getattr(self,key)
                setattr(self,key,arg)
            except AttributeError:
                print('Keyword Argument konnte nicht zugeordnet werden.')
        self.update()
    def update(self):
        '''
        '''
        self.Name = self.Name.upper()
        self.Thickness = sum(self.Thicknesses)
        self.Mass = 0
        self.Uvalue = 0
        error = False
        s_l             = 0
        a_i             = 7.7 #standard interior heat exchange coefficient 7.7 W/m?K
        a_o             = 25  #standard exterior heat exchange coefficient 25 W/m?K
        for l, (layer, d) in enumerate(zip(self.Layers, self.Thicknesses)):
                if layer in self.Materials:
                    self.Materials[layer].Use(self.Name)
                    if not np.isnan(self.Materials[layer].Conductivity):
                        #print(f'{layer} Conductivity gefunden!')
                        s_l += float(d)*3.6/self.Materials[layer].Conductivity
                    if not np.isnan(self.Materials[layer].Resistance):
                        #print(f'{layer} Resistance gefunden!')
                        s_l += 3.6*self.Materials[layer].Resistance
                    if not np.isnan(self.Materials[layer].Density):
                        #print(f'{layer} Density gefunden!')
                        self.Mass = self.Materials[layer].Density * d
                else:
                    error = True
                    #print(f'{layer} Bauteil nicht gefunden...!')
        if error:
            self.Uvalue = 0
            self.Mass = 0
        else:
            self.Uvalue =  1 / (1 / a_i + s_l + 1 / a_o)


    def createFromDB(self, Name, group):
        '''
        '''
        self.Name = str(Name).upper()
        self.Layers = list(group['Layer'].array)
        self.Thicknesses = list(group['Thickness'].array)
        self.Abs_Back = group['Abs_Back'].min()
        self.Abs_Front = group['Abs_Front'].min()
        self.H_Back = group['H_Back'].min()
        self.H_Front = group['H_Front'].min()
        self.update()
        return self

    def scale(self, factor):
        '''
        '''
        for i in range(len(self.Thicknesses)):
            self.Thicknesses[i] *= factor
    
    def get_Materials(self):
        '''
        '''
        return pd.concat([mat() for name, mat in self.Materials.items()])
    
    def changeLayerThickness(self, LayerID, newThickness):
        '''
        '''
        print(f'Ändere die Dicke des Layers {self.Layers[LayerID]} von {self.Thicknesses[LayerID]} auf {newThickness}')
        self.Thicknesses[LayerID] = newThickness
        self.update()
    def info(self):
        return pd.Series({'Name': self.Name, 'Dicke': self.Thickness, 'U-Wert': self.Uvalue, 'Masse': self.Mass})
    def __call__(self):
        cons = [{'Name':self.Name, 'Layer': self.Layers[0], 'Thickness':self.Thicknesses[0], 'Abs_Front':self.Abs_Front, 'Abs_Back':self.Abs_Back,'H_Front':self.H_Front, 'H_Back':self.H_Back}]
        if len(self.Layers) > 1:
            for layer, d in zip(self.Layers[1:], self.Thicknesses[1:]):
                cons.append({'Name':'', 'Layer': layer, 'Thickness':d, 'Abs_Front':'', 'Abs_Back':'','H_Front':'', 'H_Back':''})
        return pd.DataFrame(cons)
    
    def __repr__(self):
        return f'WALL_OBJECT <{self.Name}>: d={self.Thickness:.2f}m, U={self.Uvalue:.3f} W/(m²*K), Mass={self.Mass:.1f} kg/m²'

class KONSTRUKTION:
    def __init__(self):
        self.Materials = {}
        self.PathMaterialsDB = './data/trnDBs/bauteile.csv'
        self.PathWallsDB = './data/trnDBs/konstruktion.csv'
        self.MaterialsDB = {}
        MaterialDB = pd.read_csv(self.PathMaterialsDB, sep = ';',index_col=['Name'])
        self.MaterialsDB = {}
        for name, row in MaterialDB.iterrows():
            self.MaterialsDB[name.upper()] = LAYER().createFromDB(name, row)
        self.Walls = {}
        WallDB = pd.read_csv(self.PathWallsDB, sep=';').fillna(method='ffill')
        for name, group in WallDB.groupby('Name'):
            self.Walls[name.upper()] = WALL().createFromDB(name, group)
        self.up()

    def up(self):
        missing = []
        for name, wall in self.Walls.items():
            wall.Mass = 0
            wall.Uvalue = 0
            s_l             = 0
            a_i             = 7.7 #standard interior heat exchange coefficient 7.7 W/m?K
            a_o             = 25  #standard exterior heat exchange coefficient 25 W/m?K
            for l, (layer, d) in enumerate(zip(wall.Layers, wall.Thicknesses)):
                    if layer in self.MaterialsDB:
                        self.Materials[layer] = self.MaterialsDB[layer]
                        self.Materials[layer].Use(wall.Name)
                        if self.Materials[layer].Conductivity:
                            #print(f'{layer} Conductivity gefunden!')
                            s_l += float(d)*3.6/self.Materials[layer].Conductivity
                        if self.Materials[layer].Resistance:
                            #print(f'{layer} Resistance gefunden!')
                            s_l += 3.6*self.Materials[layer].Resistance
                        if self.Materials[layer].Density:
                            #print(f'{layer} Density gefunden!')
                            wall.Mass = self.Materials[layer].Density * d
                    else:
                        missing.append(layer)
                        #print(f'{layer} Bauteil nicht angelegt...!')
            wall.Uvalue =  1 / (1 / a_i + s_l + 1 / a_o)
            wall.Thickness = sum(wall.Thicknesses)
        self.Missing = list(set(missing))
        self.dfMats = pd.concat([mat() for name, mat in self.Materials.items()],ignore_index=True)
        self.dfWalls = pd.concat([wall() for name, wall in self.Walls.items()],ignore_index=True)
    
    def AddWall(self, wall):
        if not isinstance(wall,WALL):
            print('FEHLGESCHLAGEN! Wall muss eine WALL() Instanz ein.')
            return
        else:
            for layer in wall.Layers:
                if layer not in self.MaterialsDB:
                    print(f'ACHTUNG! {layer} nicht vorhanden. Bitte zuerst anlegen.') 
            self.Walls[wall.Name] = wall
            self.up()
            return self.Walls

    def AddMaterial(self, mat):
        if not isinstance(mat,LAYER):
            print('FEHLGESCHLAGEN! Wall muss eine MATERIAL() Instanz ein.')
            return
        else:
            self.MaterialsDB[mat.Name] = mat
            self.up()

    def scaleWall(self, WallName, factor):
        if WallName not in self.Walls:
            print('FEHLGESCHLAGEN! Wand wurde nicht gefunden. Rechtschreibung überprüfen oder Wand neu anlegen')
            return 
        else:
            print(f'Skaliere Wand um Faktor {factor}:\nAlte Eigenschaften: {self.Walls[WallName]}')
            self.Walls[WallName].scale(factor)
            self.up()
            print(f'Neue Eigenschaften: {self.Walls[WallName]}')

    def changeLayerThickness(self, WallName, LayerID, newThickness):
        if WallName not in self.Walls:
            print('FEHLGESCHLAGEN! Wand wurde nicht gefunden. Rechtschreibung überprüfen oder Wand neu anlegen')
            return 
        else:
            print(f'Alte Eigenschaften: {self.Walls[WallName]}')
            self.Walls[WallName].changeLayerThickness(LayerID, newThickness)
            self.up()
            print(f'Neue Eigenschaften: {self.Walls[WallName]}')
    
    def changeMaterialProperty(self, MatName,Value, newValue, factor=None):
        if MatName not in self.MaterialsDB:
            print('FEHLGESCHLAGEN! Wand wurde nicht gefunden. Rechtschreibung überprüfen oder Wand neu anlegen')
            return 
        else:
            try:
                print(f'Alter Wert für {Value}: {getattr(self.MaterialsDB[MatName], Value)}')
                for wallname in self.MaterialsDB[MatName].UsedInWalls:
                    print(self.Walls[wallname])
                setattr(self.MaterialsDB[MatName], Value, newValue)
                self.UpdateProperties()
                print(f'Neuer Wert für {Value}: {getattr(self.MaterialsDB[MatName], Value)}')
                for wallname in self.MaterialsDB[MatName].UsedInWalls:
                    print(self.Walls[wallname])
            except AttributeError:
                print('FEHLGESCHLAGEN! Attribut nicht gefunden...')
                return

    def export(self, path='./sim/robust-inputs', which = 'both'):
        '''
        Speichert die Datenbanken als .csv-Datei ab.

        args:
        path <str>: Ordner in den die Files exportiert werden sollen.
            default = './sim/robust-inputs'
        which <str>: Welche Datei soll exportiert werden? 'Material', 'Walls', 'both'
            default = 'both'

        returns None
        ''' 
        if which == 'Materials' or which == 'both':
            self.dfMats[['GWP','PERT','PENRT']].replace(np.NaN, 0)
            pathMats = os.path.join(path, f'RobusteMaterialien.csv')
            self.dfMats.to_csv(pathMats, sep=';', index=False)
            print(f'Materialdatenbank wurde als {pathMats} exportiert!')
        if which == 'Walls' or which == 'both':
            pathWalls = os.path.join(path, f'RobusteWaende.csv')
            self.dfWalls.to_csv(pathWalls, sep=';', index=False)
            print(f'Konstruktionsdatenbank wurde als {pathWalls} exportiert!')
    
    def exportDBs(self,which='both'):
        if which == 'Walls' or which == 'both':
            self.dfWalls.to_csv(self.PathWallsDB, index=False,sep=';')
        if which == 'Materials' or which == 'both':
            self.dfMats.to_csv(self.PathMaterialsDB, index=False,sep=';')
            
    def __repr__(self):
        return f'KONSTRUKTION COLLECTION: {len(self.Walls)} Walls, {len(self.Materials)} Materials, Not Found Materials: {self.Missing}'

class LizardWindowDB:
    def __init__(self):
        self.path = './data/trnDBs/fenster.txt'
        self._open()

    def _open(self):
        with open(self.path) as f:
            data = f.read().splitlines()
            windows = []
            for line in data:
                window = {}
                line = (line.split('='))
                info = line[0].strip()
                window['id'] = line[1]
                for value in info.split(','):
                    value = value.split('_',1)
                    try:
                        window[value[0]] = float(value[1])
                    except ValueError:
                        window[value[0]] = value[1]
                windows.append(window)
        self.df = pd.DataFrame.from_dict(windows)
        self.df = self.df.set_index(self.df.id.astype(int))
    
    def chooseBasedOnTVIS(self, TVIS):
        wind_tvis = self.df[(self.df['TVIS'] == TVIS)]
        try:
            return int(wind_tvis[wind_tvis['SHGC'] == wind_tvis.SHGC.min()].nsmallest(1, 'U'))
        except TypeError:
            return np.NaN
    def get(self, winid):
        if winid in self.df.index:
            return self.df.loc[winid]
        else:
            return np.NaN
    def __call__(self):
        return self.df

class MC_variable():
    '''
    Erstelle eine stochastisch verteilte Variable für die Monte Carlo Simulation.

    Übergebe:
    -----
        Name:   Name der Variable.
        Dist:   {'lognorm', 'norm', 'gleich', 'gumbel'}
        Sample: Stichprobe
        Mu:     Erwartungswert
        Sigma:  Standardabweichung
        bins:   Falls die Zufallsvariable diskret verteilt ist, können hier die größen definiert werden:
            int:    Anzahl an Variablen
            list(int,int,...):  Grenzen der Bins
            list(str, str, str,...): Namen der Variablen. 
        SampleRound:    Rundung der Zufallsgröße. (default = 3)
    '''

    def __init__(self, Name:str, Dist:str, nbase2:int=1, Mu=None, Sigma=None, bins=None, SampleRound=3):
        self.name = Name
        self.mu = Mu
        self.sigma = Sigma
        self.distname = Dist
        self.n_bins = None
        self._Sample = qmc.Sobol(d=1, scramble=True).random_base2(m=nbase2).reshape(-1)
        self.n = len(self._Sample)
        if isinstance(bins, int):
            self.bins = bins
            self.n = bins
            self.n_bins = bins
            self.labels = range(bins)
        elif isinstance(bins, list):
            if all(isinstance(item, str) for item in bins):
                self.bins = len(bins)
                self.labels = bins
                self.n_bins = self.bins
            if all(isinstance(item, (float, int)) for item in bins):
                self.bins = bins
                self.labels = bins[1:]
                self.n_bins = len(self.bins)
        else:
            self.bins = None
            self.labels = None
        
        self.legend_safeRBs = []
        self.accuracy = SampleRound
        if isinstance(SampleRound, int):
            r_h = 1
        else:
            r_h = SampleRound * 10
            SampleRound = 1
        

        if Dist == 'lognorm':
            self.dist = lognorm(s=self.sigma, scale=self.mu)
            self.sample = np.multiply(np.divide(self.dist.ppf(self._Sample), r_h).round(SampleRound), r_h)
            self._x = self.sample
        elif Dist == 'norm':
            self.dist = norm(scale=self.sigma, loc=self.mu)
            self.sample = np.multiply(np.divide(self.dist.ppf(self._Sample), r_h).round(SampleRound), r_h)
            self._x = self.sample
        elif Dist == 'gleich':
            self.dist = uniform(scale=self.n_bins)
            self._x = self.dist.ppf(self._Sample)
            self.sample = np.array(pd.cut(self._x, bins=self.bins, labels=self.labels))
        elif Dist == 'discrete':
            self.dist = st.randint(0,self.bins)
            self._x = self.dist.ppf(self._Sample)
            self.sample = np.array(pd.cut(self._x, bins=self.bins, labels=self.labels))
        elif Dist == 'gumbel':
            self.dist = st.gumbel_l(scale=self.sigma, loc=self.mu)
            self.sample = np.multiply(np.divide(self.dist.ppf(self._Sample), r_h).round(SampleRound), r_h)
            self._x = self.sample
        else:
            raise ValueError('False Verteilung übergeben. Verteilung muss ["lognorm", "norm" oder "gleich"] sein')

        self.info = pd.DataFrame(index = [self.name],data={'Mu': self.mu, 'Sigma': self.sigma, 'Verteilung':self.distname})

    def __repr__(self):
        return f'SIMULATIONS-VARIABLE <{self.name}> | Verteilung {self.distname}'
    
    def plot(self, sample = True, dist = True, annotate=False, ax=None, color=None):
        '''
        Erstelle einen Verteilungsplot für eine Zufallsvariable.

        Übergebe:
        -----
        '''
        if ax: 
            self.ax = ax
            self.fig = ax.get_figure()
        else:
            self.fig, self.ax = plt.subplots()

        x = np.linspace(self.dist.ppf(0.01),
                self.dist.ppf(0.99), 100)
        self.ax.plot(x, self.dist.pdf(x), linestyle='dashed',marker='None', color='k')

        if sample:
            self.ax.plot(self._x, self.dist.pdf(self._x), mfc='none', linestyle='none', label='Sample', marker='o', alpha=1, color='k')
            if self.distname == 'gleich':
                binkws = dict(bins = range(self.bins+1))
            else:
                binkws = dict(bins = np.arange(np.floor(np.min(self._x))-self.accuracy, np.ceil(np.max(self._x))+self.accuracy, step=1))
            self.ax.hist(self._x, density=True, color='k', fill=False, **binkws)
            self.ax.set(xlabel=self.name, ylabel = 'rel. Häufigkeit')
            if self.labels is not None:
                self.ax.set_xticks([i+.5 for i in range(len(self.labels))], self.labels)

        if annotate:
            if ax != None:
                self.ax.set_title('Stichprobe mit Wahrscheinlichkeitsverteilung', loc='left')
            textbox = ''
            if sample and dist:
                if self.distname in ['norm', 'lognorm']:
                    textbox = f'{self.name} ({self.distname})\n' r'$\mu:$' + f'{self.mu:.0f}\n' + r'$\sigma:$' + f'{self.sigma:.1f}\nn: {self.n}'
                elif self.distname == 'gleich':
                    textbox = f'{self.name} ({self.distname})\n' f'bins: {self.labels}\nn: {self.n}'
            elif dist and not sample:
                if self.distname in ['norm', 'lognorm']:
                    textbox = f'{self.name} ({self.distname})\n' r'$\mu:$' + f'{self.mu:.0f}\n' + r'$\sigma:$' + f'{self.sigma:.1f}'
                elif self.distname == 'gleich':
                    textbox = f'{self.name} ({self.distname})'            
            
            if textbox:
                self.legend_safeRBs.append(Patch(facecolor=color, label=textbox))

            self.ax.legend(handles=self.legend_safeRBs, loc='best')
        return self.ax
        
class Bauweise:
    '''
    Definition einer TRNLIZARD-Bauweise. 

    Die einzelnen Attribute enthalten die Namen der entsprechenden Bauteile in der TRNLIZARD Library
    '''
    def __init__(self, **kwargs):
        self.Name = ''
        self.AW = ''
        self.IW_H = ''
        self.IW_L = ''
        self.DA = ''
        self.BO = ''
        self.FBH = ''
        self.WindowID = ''
        self.THB = 0.05
        self.ActiveLayer = int(isinstance(self.FBH, WALL))

        for key, arg in kwargs.items():
            try:
                getattr(self,key)
                setattr(self,key,arg)
            except AttributeError:
                print('Keyword Argument konnte nicht zugeordnet werden.')
        self.up()
    def up(self):
        self.info = pd.concat([getattr(self, wall).info() for wall in ['AW', 'IW_L','IW_H', 'DA', 'BO']], axis=1).T.round(2)
        self.info['THB'] = self.THB
        self.window = LizardWindowDB().get(self.WindowID)[['U', 'SHGC', 'TVIS']].rename('Fenster').to_frame()
    
    def __repr__(self):
        return f'BAUWEISE <{self.Name}>'

class Variant:
    ''''
    Simulationsvariante.
    In dieser Klasse werden alle Informationen die als Variablen in die Simulation einfliesen sollen definiert. 
    
    Inputs können einzelne Werte und/oder eine <Bauweise>-Klasse sein.

    Im Attribut VariantInformation werden die Variablen gesammelt, die an TRNLizard übergeben werden sollen. 
    Um eine neue Variable anzulegen, muss ein Attribut (self.<Attribut>) erstellt werden und deren Name in die VariantInformation Liste eingefügt werdne.
    
    Wenn die Variante 
    
    '''
    def __init__(self, Name='Template', **kwargs):

        # Bauweise
        self.Name           = Name          #str
        self.bui            = ''            #str
        self.IntWall_H      = ''            #str
        self.IntWall_L      = ''            #str
        self.ExtWall        = ''            #str
        self.Floor          = ''            #str
        self.Ceiling        = ''            #str
        self.Laibung        = ''            #float
        self.FBH            = 'FBH_DUMMY'   #str
        self.THB            = 0.05          #float
        self.ActiveLayer    = 0             #bool
        self.WindowID       = ''            #int

        # Geometrie
        self.WindowRatio    = 0.239089      #float
        self.BoxWidth      = 3.4            #float
        self.BoxLength     = 5.26           #float
        self.BoxHeight     = 3.1            #float
        self.Airnodes      = 3              #int
        self.RRatioWidth    = 0.708         #float
        self.RRatioDepth    = 0.384         #float
        self.Rotation       = -97           #int
        self.WindowUV       = (1.4,1.8)     #tuple(float, float)
        self.WindowPosition = 0.692         #float
        self.WindowOrientation = 'W'        #str: N, O, S, W

        # Nutzung
        self.PersonsPerAppartment  = 1 # [-]
        self.ACR_inf        =   0     # 1/h
        self.Tset           =   21    # °C
        self.IntGain        =   6     # W/m²
        self.WeatherFile    =   'MUC_2020.epw'    
        self.FensterAuf     =   4     # K über Heizgrenztemperatur
        self.FensterZu      =   2     # K über Heizgrenztemperatur
        self.LüftenProTag   =   '4x15min' # Wie oft wird pro Tag das Fenster geöffnet
        self.FensterSignal  =   'None_0'
        self.FensterModel   =   0
        # self.VariantInformation = sorted(['bui', 'IntWall_L', 'IntWall_H', 'FBH',
        #  'ExtWall','WindowRatio', 'BoxWidth', 'BoxLength',
        #   'BoxHeight', 'Airnodes', 'RRatioWidth', 'RRatioDepth', 'PersonsPerAppartment',
        #   'Rotation', 'WindowUV', 'Laibung', 'THB', 'WindowPosition', 'WindowOrientation','FensterSignal', 'FensterModel',
        #    'ActiveLayer', 'Floor', 'Ceiling', 'WindowID', 'ACR_inf', 'Tset', 'IntGain', 'WeatherFile','SV_ON', 'SV_T_ON','SV_T_OFF'])
        self.VariantInformation = sorted(['bui', 'IntWall_L', 'IntWall_H', 'FBH',
         'ExtWall', 'Laibung', 'THB','FensterSignal', 'FensterModel',
           'ActiveLayer', 'Floor', 'Ceiling', 'WindowID', 'ACR_inf', 'Tset', 'IntGain', 'WeatherFile','SV_ON', 'SV_T_ON','SV_T_OFF'])

        up = False
        for key, item in kwargs.items():
            if hasattr(self,key): 
                setattr(self, key, item)
                up = True
            else:
                print(f'Achtung! Attribut {key} wurde nicht gefunden.')
        if up:
            self.up()

    def up(self):
        '''
        Update die Inputsensitiven Variablen (Temperaturabhängige Lüftung, Bauweise)
        '''
        errors = []
        self.FensterModel   = self.FensterSignal.split('_')[0]
        if self.FensterModel not in ['MC', 'LogReg']:
            self.SV_T_ON        = self.Tset + self.FensterAuf
            self.SV_T_OFF       = self.Tset + self.FensterZu
            self.SV_ON          = 6
        else:
            self.SV_T_ON        = -20
            self.SV_T_OFF       = -20
            self.SV_ON          = -20
        if isinstance(self.bui, Bauweise):
            self.setBUI(self.bui)
        for key in self.VariantInformation:
            if getattr(self, key) == '':
                errors.append(key)
        self.Dict = {key: self.__dict__[key] for key in self.VariantInformation}
        if len(errors) > 0:
            print(f'Variante {self.Name}: {errors} wurden nicht korrekt definiert!')
            self.Dict = {}

    def setBUI(self, bauweise, up=True):
        '''
        Befülle die Variable mit den Informationen aus einer <Bauweise>-Klasse
        '''
        self.bui = bauweise.Name
        self.IntWall_H = bauweise.IW_H.Name
        self.IntWall_L = bauweise.IW_L.Name
        self.ExtWall = bauweise.AW.Name
        self.Floor = bauweise.BO.Name
        self.Ceiling = bauweise.DA.Name
        self.Laibung = bauweise.AW.Thickness
        self.WindowID = bauweise.WindowID
        self.ActiveLayer = bauweise.ActiveLayer
        if self.ActiveLayer == 1:
            self.FBH = bauweise.FBH.Name
        self.THB = bauweise.THB

    def fillFromDict(self, vardict):
        '''
        Befülle eine Variante mit den Variablen aus einem Dict.
        '''
        if isinstance(vardict, dict):
            for key, item in vardict.items():
                if isinstance(item, Bauweise):
                    self.setBUI(item, up=False)
                elif not hasattr(self,key): print(f'Achtung! Attribut {key} wurde nicht gefunden und neu erstellt.')
                setattr(self, key, item)
        self.up()
        return self

    def __call__(self):
        return pd.DataFrame([self.Dict], index=[self.Name])


class VariantCollection:
    def __init__(self, Name = 'Template'):
        self.Name = Name
        self.Simulationen = []
        self.VariableInfos = {}
        self.df = pd.DataFrame(columns = Variant().VariantInformation)
        self.resultpath = './sim/robust-inputs'

    def __call__(self):
        return self.df

    def AddSingleVariant(self, var:Variant):
        self.df = pd.concat([self.frame, var()])
        self.df.index = [f'{self.Name}_{i}' for i in range(self.df.shape[0])]
        self.df.index.set_names(['Name'],inplace=True)
        
    def MonteCarloVariants(self, SimName:str, Varianten:dict=None, unsichereRB:dict=None):
        self.RB = pd.DataFrame([{key: el for key, el in zip(Varianten, safeRB)} for safeRB in itertools.product(*Varianten.values())])
        
        if unsichereRB is not None:
            self.VariableInfos.update(unsichereRB)
            unsafeRB = pd.DataFrame(pd.Series(unsichereRB, dtype='object').apply(lambda x: x.sample).to_list(), index = unsichereRB.keys()).T
            self.RB = self.RB.merge(unsafeRB, how='cross')

        self.Simulationen.extend(self.RB.apply(lambda x: Variant(Name=f'{SimName}_{x.name}', **x.to_dict()), axis=1).to_list())
        self.df = pd.concat([item() for item in self.Simulationen])
        return self.df

    def open(self,dest='./sim/robust-inputs/VariantCollection.csv'):
        self.df = pd.read_csv(dest, sep=';', index_col=['Name'])
        return self.df

    def export(self, *args, sample=None, **kwargs):
        self.df.index = self.df.index.set_names(['Name'])
        m = []
        for key, item in kwargs.items():
            if key in self.df.columns:
                m.append(self.df[key] == item)
        if len(m) > 0:
            m = np.array(m).T.any(axis=1)
            if sample:
                self.df.loc[m,:].groupby(list(args)).sample(sample).round(3).to_csv(os.path.join(self.resultpath, 'VariantCollection.csv'),index=True, sep=';')
            else:
                self.df.loc[m,:].round(3).to_csv(os.path.join(self.resultpath, 'VariantCollection.csv'),index=True, sep=';')
        else:
            if sample:
                self.df.groupby(list(args)).sample(sample).round(3).to_csv(os.path.join(self.resultpath, 'VariantCollection.csv'),index=True, sep=';')
            else:
                self.df.round(3).to_csv(os.path.join(self.resultpath, 'VariantCollection.csv'),index=True, sep=';')

        if hasattr(self, 'variable_infos'):
            info = [variable.info for name, variable in self.variable_infos.items()]
            pd.concat(info,axis=0).to_csv(os.path.join(self.resultpath, 'VariableDistributions.csv'))
            print(f'VariantCollection wurde unter {self.resultpath} gespeichert!')
    

def read_prn(path, index='HoY'):
    with open(path,'r') as f:
        f = f.read().splitlines()
        data = []
        for line in f:
            line = line.split()
            data.append(line)
        units = data[0]
        headers = data[1]
        for j, line in enumerate(data):
            if len(line) == 0:
                data = data[2:j]
                break
    if headers[0] == 'Period' and (len(units) == len(headers)-1):
        units = [''] + [f'[{unit}]' for unit in units]
        headers = [f'{name} {unit}'.strip() for name, unit in zip(headers, units)]
    elif len(units) == len(headers):
        headers = [f'{name} [{unit}]'.strip() for name, unit in zip(headers, units)]
        print('Keine Zeitleiste')
    else:
        print('Header und Einheiten stimmen nicht überein..')

    df = pd.DataFrame({head: dat for head, dat in zip(headers,list(map(list, zip(*data))))})
    for col in df:
        df[col] = pd.to_numeric(df[col],errors='coerce')
    df = df.set_index('Period')
    if index == 'HoY':
        return df
    elif index == 'dt':
        return PeriodToDatetimeIndex(df)
    else:
        raise AttributeError('Index must be "dt" for Datetime or "HoY" for Period Index.')
def merge_prns(prns):
    df = pd.concat(prns,axis=1)
    df = df.loc[:,~df.columns.duplicated()]
        
    idx = []
    for col in df.columns:
        #pd.to_numeric(df[col],errors='coerce')
        col = re.split(r'(?<!MF)_A_',col)
        id1 = ''
        id2 = ''
        if len(col) != 2:
            found = False
            for item in col:
                r = re.search(r'\[([A-Za-z0-9_]+)\]',item)
                if r:
                    if r[1] == 'UserEquation':
                        found = True
                        id1 = 'UserEquation'
                        id2 = item.split()[0]
                        #print(f'UserEquation found! {id1=}{id2=}')
                if item.startswith('MF_'):
                    found = True
                    id1 = 'Coupling'
                    id2 = item
                    #print(f'coupling found! {id1=}{id2=}')
                if 'amb' in item:
                    found = True
                    id1 = 'Amb'
                    id2 = item
                    #print(f'weatherdata found!  {id1=}{id2=}')
            if found == False:
                id2 = col[0]
                id1 = ''
                print(f'unmatched column: {col}', end='\r')
        else:
            id2 = f'{col[0]} {col[1].split()[1]}'
            id1 = col[1].split()[0]
            #print(f'{id1=}{id2=}')
        idx.append((id2,id1))

    if len(df.columns) == len(idx):
        df.columns = pd.MultiIndex.from_tuples(idx)
    else:
        return

    df.sort_index(axis=1,inplace=True)
    df.columns.rename(['Value', 'Airnode'],inplace=True)
    return df

class BALREADER():
    def __init__(self, path, area=None):
        self.path = path
        self.area = area

def read_summaryBAL(path, ref_floorarea=None):
    if os.path.isfile(path):
        files = [path]
    elif os.path.isdir(path):
        files = []
        refareas = []
        for variant in os.listdir(path):
            folder = os.path.join(path, variant)
            balfile = [os.path.join(folder, file) for file in os.listdir(folder) if file.endswith('.BAL')]
            if len(balfile) > 0:
                files.extend(balfile)

                if ref_floorarea is None:
                    refareas.append(0)
                elif isinstance(ref_floorarea, (str, int, float)):
                    refareas.append(ref_floorarea)
                elif isinstance(ref_floorarea, (pd.Series, dict)):
                    try:
                        refareas.append(ref_floorarea[variant])
                    except KeyError:
                        refareas.append(0)
                else:
                    refareas.append(0)
        if len(files) != len(refareas):
            raise ValueError
    dfs = []
    for file, area in zip(files, refareas):
        if os.path.exists(file):
            name = (os.path.basename(os.path.dirname(file)))
            with open(file,'r') as f:
                f = f.read().splitlines()
                for l, line in enumerate(f):
                    if line.startswith('  Zonenr'):
                        headers = line
                    if 'Energy balance for sum of all zone' in line:
                        sums = f[l+1]
            varbal = {}
            if area > 0:
                print('BAL-Einheit: kWh/m²*a',end='\r',flush=True)
                for head, value in zip(headers.split(), sums.split()):
                        head = ''.join(filter(str.isalnum, head))
                        varbal[head] = round(float(value)/3600/area,2)
            else:
                print('BAL-Einheit: kWh/a',end='\r',flush=True)
                for head, value in zip(headers.split(), sums.split()):
                    head = ''.join(filter(str.isalnum, head))
                    varbal[head] = round(float(value)/3600,2)
            dfs.append(pd.Series(varbal, name=name))
    df = pd.concat(dfs, axis=1).T

    return df


def PeriodToDatetimeIndex(df, year:int=None):
    def leap_year(y):
        if y % 400 == 0:
            return True
        if y % 100 == 0:
            return False
        if y % 4 == 0:
            return True
        else:
            return False

    if year is None:
        if df.shape[0]%8760 == 0:
            step = df.shape[0]/8760
            year = 2022
        elif df.shape[0]%8784 == 0:
            step = df.shape[0]/8784
            year = 2020
        else:
            raise ValueError('Umformung funktioniert nur mit einer Jahressimulation.')
    else:
        if leap_year(year):
            if df.shape[0]%8784 == 0:
                step = df.shape[0]/8784
                pass
            else:
                raise ValueError(f'Achtung das Jahr {year} ist ein Schaltjahr. Der Datensatz hat jedoch nur 365 Tage.')
        else:
            step = df.shape[0]/8760
    freq = f'{int(60/step)}min'
    _df = df.copy()
    _df.index = pd.DatetimeIndex(pd.date_range(pd.Timestamp(year, 1, 1 ), pd.Timestamp(year+1, 1,1), freq=freq, inclusive='left'))
    _df.index = _df.index.set_names(['Datetime'])
    return _df

def HOYtoDatetime(hoy, year):
    '''
    Wandle HourOfYear Daten in ein Datetime Objekt um.

    Args:
    -----
    hoy: int
            Hour of Year
    year: int
            Jahr des Datensatzes
        
    Returns: 
    ----
    int
    '''
    return dt.datetime(year,1,1,0,0) + dt.timedelta(hoy)

def DatetimeToPeriodIndex(df):
    if len(df.index.year.unique()) == 1:
        year = df.index.year.unique()[0]
    else:
        print('Achtung! Der Datensatz erstreckt sich über ein Jahr! HOY-Werte sind nur innerhalb des gleichen Jahres sinnvoll.')
        return
    df.index = (((df.index.to_series()-dt.datetime(year,1,1,0,0, tzinfo=tz)).dt.total_seconds()/60/60)+1).astype('int')
    df.index = df.index.set_names(['Period'])
    return df

# ======== copy Simulations from LS-Workstation ==========
def copy_results_from_WS(SIM_FOLDER = r'W:\Thesis', MODEL_FOLDER = r'.\_db\simulations', simname=None):
    """
    Diese Funktion synchronisiert die Simulationsergebnisse aus dem Ordner 'SIM_FOLDER' mit dem ordner 'MODEL_FOLDER'.

    ARGS:
    -----
        SIM_FOLDER:     
            Ordner in dem neue Simulationen abgespeichert werden.   (default = W:\\Thesis)
        MODEL_FOLDER:   
            Ordner in dem die Simulationen zur Analyse abgelegt werden. (default = .\\_db\\simulations)
        simname:        
            Hier kann eine bestimmte Simulation vorab bestimmt werden. Wenn keine Auswahl getroffen wird, wird die zu synchronisierende Simulation abgefragt.
    """
    if not simname:
        simulations = {i+1: sim for i, sim in enumerate(os.listdir(SIM_FOLDER))}
        simID = int(input(f'Ich habe folgende Simulationen gefunden: {simulations}\nWelche soll geöffnet werden? 0 importiert nichts.'))
        if simID == 0:
            print('Import übersprungen...')
            return r'.\sim\Model'
        simname = simulations[simID]
    sim = os.path.join(SIM_FOLDER,simname,'Model')
    src = os.path.join(sim,'_results')
    dst = os.path.join(MODEL_FOLDER,simname,'_results')
    try:
        shutil.copytree(src,dst)
    except FileExistsError:
        for file in os.listdir(src):
            if os.path.isdir(os.path.join(src,file)):
                subdir = os.path.join(src,file)
                subdst = os.path.join(dst,file)
                if not os.path.exists(subdst): os.makedirs(subdst)
                for file2 in os.listdir(subdir):
                    if os.path.exists(os.path.join(subdst,file2)):
                        if os.path.getmtime(os.path.join(subdir,file2)) > os.path.getmtime(os.path.join(subdst,file2)):
                            shutil.copy2(os.path.join(subdir,file2),os.path.join(subdst,file2))
                            print('{} updated!'.format(file2))
                        else:
                            print('{} already exists!'.format(file2))
                    else:
                        shutil.copy2(os.path.join(subdir,file2),os.path.join(subdst,file2))
            elif os.path.exists(os.path.join(dst,file)):
                if os.path.getmtime(os.path.join(src,file)) > os.path.getmtime(os.path.join(dst,file)):
                    shutil.copy2(os.path.join(src,file),os.path.join(dst,file))
                    print('{} updated!'.format(file))
                else:
                    print('{} already exists!'.format(file))
            else:
                shutil.copy2(os.path.join(src,file),os.path.join(dst,file))
                print('{} created!'.format(file))

    nv = len(os.listdir(sim))
    for vi, var in enumerate(os.listdir(sim)):
        path = os.path.join(sim, var)
        collect = []
        for fn in os.listdir(path):
            if fn.endswith('.BAL'):
                collect.append(fn)
            elif fn.endswith('infos.log'):
                collect.append(fn)
            elif fn == 'Results':
                collect.append(fn)

        for file in collect:
            src = os.path.join(path, file)
            variant = os.path.join(MODEL_FOLDER,simname,var)
            if not os.path.exists(variant):
                os.makedirs(variant)
            dst = os.path.join(MODEL_FOLDER,simname,var, file)
            if os.path.exists(dst) and os.path.getmtime(dst) > os.path.getmtime(src):
                    pass
            else:
                try:
                    shutil.copytree(src, dst)
                except OSError as exc: # python >2.5
                    if exc.errno in (errno.ENOTDIR, errno.EINVAL):
                        shutil.copy(src, dst)
        running_bar(vi,nv)
    return os.path.join(MODEL_FOLDER, simname)

class Value109:
    '''
    Diese Klasse beinhaltet eine Spalte eines .109-Wetterdatensatzes.
    
    Funktionen:
        printInfoString()
        
    
    Diese Funktion ist im Rahmen meiner Masterarbeit zum Thema 'Robust Bauen' entstanden.

    Author:
    Roman Ficht
    Version 0.1
    Datum 16.03.2022
    '''
    def __init__(self):
        self.var = 'IBEAM_M'
        self.col = 2
        self.interp = 0
        self.add = 0
        self.mult = 1
        self.samp = 0
        self.docstring = '!...to get radiation in W/m2'
        self.data = []

    def printInfoString(self):
        return f'<var>   {self.var} <col>  {self.col}  <interp> {self.interp}  <add>  {self.add}  <mult>  {self.mult}  <samp>   {self.samp}'
    def update(self, df):
        self.data = df
    def __repr__(self):
        return f'109-Variable <{self.var}>'
    def __call__(self):
        return self.data

class weather109:
    '''
    
    Diese Klasse kann Wetterdatensätze vom Dateiformat .109 einlesen, bearbeiten und exportieren
    
    Funktionen:

        read_file(path):
            Liest eine .109-Wetterdatei ein und 
    
    Diese Funktion ist im Rahmen meiner Masterarbeit zum Thema 'Robust Bauen' entstanden.

    Author:
    Roman Ficht
    Version 0.1
    Datum 16.03.2022

    '''
    def __init__(self):
        self.name = 'WeatherFile'
        self.userdefined = ''
        self.longitude = 0 # east of greenwich: negative; (shift for use of TYPE 16: -2.50   )
        self.latitude = 0
        self.gmt = 0 # time shift from GMT, east: positive (hours), solar = solartime
        self.interval = 1 # Data file time interval between consecutive lines
        self.firsttime = 1 # Time corresponding to first data line (hours)
        self.Variables = {}
        self.year = ''
    
    def read_file(self, path, year=2010):
        self.name = os.path.basename(path).rsplit('.',1)[0]
        self.path = os.path.dirname(path)
        self.year = year
        with open(path, mode='r', encoding='latin-1') as data:
            data = data.read().splitlines()
            lines = []
            for l, line in enumerate(data):
                if '!' in line:
                    for p, part in enumerate(line.split()):
                        if '!' in part:
                            line = line.split()[:p]
                else:
                    line = line.split()
                if len(line) > 0:
                    lines.append(line)
                    find = re.search(r'<([^\)\s]+)>', line[0])
                    if find:
                        if hasattr(self,find[1]):
                            if len(line) == 2:
                                setattr(self, find[1], line[1])
                            if len(line) == 1:
                                setattr(self, find[1], '')
                        if find[1] == 'var':
                            v = Value109()
                            for i, item in enumerate(line):
                                find2 = re.search(r'<([^\)\s]+)>', item)
                                if find2:
                                    if hasattr(v,find2[1]):
                                        setattr(v,find2[1], line[i+1])
                            self.Variables[int(v.col)] = v
                        if find[1] == 'data':
                            data = l
        data = lines[data+1:]
        data = list(map(list, zip(*data)))
        if len(data) == len(self.Variables) + 1:
            self.Period = data[0]
            for key, item in self.Variables.items():
                item.data = pd.Series(list(map(float, data[key-1])), self.Period, name=item.var)

        self.df = pd.concat([v() for col, v in self.Variables.items()],axis=1)
        self.df.index = pd.to_numeric(self.df.index, downcast='integer')
    def set_data(self, df):
        pass
    def _writeheader(self): 
        data = [f'<userdefined>   {self.userdefined}', f'<longitude>   {self.longitude}',f'<latitude>   {self.latitude}',f'<gmt>   {self.gmt}',f'<interval>   {self.interval}',f'<firsttime>   {self.firsttime}']
        return data
    def _writedataheader(self):
        head = []
        for key, item in self.Variables.items():
            head.append(item.printInfoString())
        return head
    def _writedata(self):
        data = ['<data>']
        for i, row in self.df.iterrows():
            row = [i] + row.to_list()
            data.append('   '.join(map(str,row)))
        return data
    def export(self,dest=None):
        if dest is None:
            dest = self.path
        data = self._writeheader() + self._writedataheader() + self._writedata()
        data = '\n'.join(data)
        self.filename = os.path.join(dest, f'{self.name}.109')
        with open(self.filename, 'w+') as f:
            f.write(data)
        print(f'Wetterdatei {self.filename} wurde erfolgreich gespeichert"')

class epw():
    """A class which represents an EnergyPlus weather (epw) file
    """
    
    def __init__(self):
        self.headers={}
        self.df=pd.DataFrame()
            
    
    def read(self,fp):
        """Reads an epw file 
        
        Arguments:
            - fp (str): the file path of the epw file   
        
        """
        
        self.headers=self._read_headers(fp)
        self.df=self._read_data(fp)
                
        
    def _read_headers(self,fp):
        """Reads the headers of an epw file
        
        Arguments:
            - fp (str): the file path of the epw file   
            
        Return value:
            - d (dict): a dictionary containing the header rows 
            
        """
        
        d={}
        with open(fp, newline='') as csvfile:
            csvreader = csv.reader(csvfile, delimiter=',', quotechar='"')
            for row in csvreader:
                if row[0].isdigit():
                    break
                else:
                    d[row[0]]=row[1:]
        return d
    
    
    def _read_data(self,fp):
        """Reads the climate data of an epw file
        
        Arguments:
            - fp (str): the file path of the epw file   
            
        Return value:
            - df (pd.DataFrame): a DataFrame comtaining the climate data
            
        """
        
        names=['Year',
               'Month',
               'Day',
               'Hour',
               'Minute',
               'Data Source and Uncertainty Flags',
               'Dry Bulb Temperature',
               'Dew Point Temperature',
               'Relative Humidity',
               'Atmospheric Station Pressure',
               'Extraterrestrial Horizontal Radiation',
               'Extraterrestrial Direct Normal Radiation',
               'Horizontal Infrared Radiation Intensity',
               'Global Horizontal Radiation',
               'Direct Normal Radiation',
               'Diffuse Horizontal Radiation',
               'Global Horizontal Illuminance',
               'Direct Normal Illuminance',
               'Diffuse Horizontal Illuminance',
               'Zenith Luminance',
               'Wind Direction',
               'Wind Speed',
               'Total Sky Cover',
               'Opaque Sky Cover (used if Horizontal IR Intensity missing)',
               'Visibility',
               'Ceiling Height',
               'Present Weather Observation',
               'Present Weather Codes',
               'Precipitable Water',
               'Aerosol Optical Depth',
               'Snow Depth',
               'Days Since Last Snowfall',
               'Albedo',
               'Liquid Precipitation Depth',
               'Liquid Precipitation Quantity']
        
        first_row=self._first_row_with_climate_data(fp)
        df=pd.read_csv(fp,
                       skiprows=first_row,
                       header=None,
                       names=names)
        return df
        
        
    def _first_row_with_climate_data(self,fp):
        """Finds the first row with the climate data of an epw file
        
        Arguments:
            - fp (str): the file path of the epw file   
            
        Return value:
            - i (int): the row number
            
        """
        
        with open(fp, newline='') as csvfile:
            csvreader = csv.reader(csvfile, delimiter=',', quotechar='"')
            for i,row in enumerate(csvreader):
                if row[0].isdigit():
                    break
        return i
        
        
    def write(self,fp):
        """Writes an epw file 
        
        Arguments:
            - fp (str): the file path of the new epw file   
        
        """
        
        with open(fp, 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile, delimiter=',',
                                    quotechar='"', quoting=csv.QUOTE_MINIMAL)
            for k,v in self.headers.items():
                csvwriter.writerow([k]+v)
            for row in self.dataframe.itertuples(index= False):
                csvwriter.writerow(i for i in row)

class SIMRESULTS():
    def __init__(self, path, focus_variables = [], VarCol=None):
        self.path = path
        try:
            if VarCol is None:
                self.VariantCollection = VariantCollection().open(os.path.join(self.path, '_results', 'VariantCollection.csv'))
            elif isinstance(VarCol,VariantCollection):
                self.VariantCollection = VarCol()
            else:
                raise FileNotFoundError
            self.variants = list(self.VariantCollection.index)
            self.n_variants = len(self.variants)
        except FileNotFoundError:
            if os.path.isdir(self.path):
                if len([file for file in os.listdir(self.path) if file.endswith('.BAL')]) > 0:
                    balfound = True
                if os.path.isdir(os.path.join(self.path, 'Results')):
                    if len([file for file in os.listdir(self.path) if file.endswith('.PRN')]) > 0:
                        prnfound = True
            if balfound or prnfound:
                self.variants = [os.path.basename(self.path)]
                self.n_variants = 1
                self.path = os.path.dirname(self.path)
                self.VariantCollection = pd.DataFrame()
        self.Name = [*{var.split('_')[0] for var in self.variants}]
        print(self.Name[0])
        self._unmatched = []
        self._prn_not_found = []
        self._loadPRN()
        self.PRN = self.PRN.join(pd.concat({'ACR_SV': self.PRN.filter(like='ACR_tot').droplevel(1, axis=1)-self.PRN.filter(like='ACR_inf').droplevel(1, axis=1)}, axis=1).swaplevel(0,1, axis=1)).sort_index(axis=1)
        self.variables = list(self.PRN.columns.get_level_values(1).unique())
        self.area = self.PRN.filter(like='Area').max()
        self.ref_area = self.area.groupby(level=0).sum()

        self._bal_not_found = []
        self._read_BAL()

        self.focus = focus_variables
        self._getKelvinstunden()
        self._createSummary()
        if len(self._prn_not_found) > 0:
            print(f'Keine PRN-Files gefunden für Variante: {self._prn_not_found}')
        if len(self._bal_not_found) > 0:
            print(f'Keine BAL-Files gefunden für Variante: {self._bal_not_found}')
        return
    
    def __call__(self):
        return self.Summary

    def getValue(self, name, **kwargs):
        if isinstance(name, str):
            if name in self.variables:
                name = [name]
            else:
                name = []
        else:
            name = [n for n in name if n in self.variables]
        _df = self.PRN.loc[:, np.array([self.PRN.columns.get_level_values(lvl).isin(name) for lvl in range(self.PRN.columns.nlevels)]).T.any(axis=1)]
        drplvl = []
        m=[]
        for key, item in kwargs.items():
            if isinstance(item, str):
                item=[item]
            m = _df.columns.get_level_values(key).isin(item)
            drplvl.append(key)

        if len(m) == 0:
            return _df
        else:
            return _df.loc[:,m].droplevel(drplvl, axis=1)

    def _createSummary(self):
        if self.VariantCollection.shape > (0,0):
            self.Summary = pd.merge(self.VariantCollection[self.focus], self.BAL.replace(0,np.NaN).dropna(axis=1,how='all'), left_index=True, right_index=True)
        else:
            self.Summary = self.BAL.replace(0,np.NaN).dropna(axis=1,how='all')
        self.Summary = self.Summary.join(self.Kelvinstunden.groupby(level=[0,2], axis=1).median().sum().unstack().round())
        self.Summary['QHEAT'].fillna(self.PRN.filter(like='Q_tot_ht').sum().droplevel(1).mul(self.area.droplevel(1)).groupby(level=0).sum().div(self.area.groupby(level=0).sum()).div(1000).round(2), inplace=True)
        try:
            self.Summary['kind'] = self.Summary.FensterSignal.apply(lambda x: x.split('_')[0])
        except AttributeError:
            pass
        if len(self.Summary.index) == self.n_variants == len(self.PRN.columns.get_level_values(0).unique()):
            print('Alle Varianten vollständig importiert')
        else:
            print('Achtung! Es wurden nicht alle gefundenen Varianten vollständig importiert')

    def _getKelvinstunden(self):
        tamb_g24 = self.PRN.filter(like='Tamb_g24').groupby(level=[0], axis=1).max()
        tamb = self.PRN.filter(like='Tamb [C]').groupby(level=[0], axis=1).max()
        kh = {}
        for (vari, room), group in self.PRN.filter(like='Top').groupby(level=[0,2], axis=1).max().iteritems():
            kh[vari, room] = KelvinstundenEN(group, Tamb_g24=tamb_g24[vari]).add_suffix('_EN').join(KelvinstundenNA(group, Tamb=tamb[vari]).add_suffix('_NA'))
        
        self.Kelvinstunden = pd.concat(kh, axis=1)
        self.PRN = self.PRN.join(self.Kelvinstunden.reorder_levels([0,2,1], 1)).sort_index(axis=1)

    def _read_BAL(self):
        dfs = []
        for variant in self.variants:
            folder = os.path.join(self.path, variant)
            if not os.path.isdir(folder):
                self._bal_not_found.append(variant)
                continue
            file = [os.path.join(folder, file) for file in os.listdir(folder) if file.endswith('.BAL')]
            if len(file) > 0: 
                file = file[0]
            else: 
                self._bal_not_found.append(variant)
                continue
            if os.path.exists(file):
                with open(file,'r') as f:
                    f = f.read().splitlines()
                    for l, line in enumerate(f):
                        if line.startswith('  Zonenr'):
                            headers = line
                        if 'Energy balance for sum of all zone' in line:
                            sums = f[l+1]
                varbal = {}
                if self.ref_area[variant] > 0:
                    print('BAL-Einheit: kWh/m²*a',end='\r',flush=True)
                    for head, value in zip(headers.split(), sums.split()):
                            head = ''.join(filter(str.isalnum, head))
                            varbal[head] = round(float(value)/3600/self.ref_area[variant],2)
                else:
                    print('BAL-Einheit: kWh/a',end='\r',flush=True)
                    for head, value in zip(headers.split(), sums.split()):
                        head = ''.join(filter(str.isalnum, head))
                        varbal[head] = round(float(value)/3600,2)
                dfs.append(pd.Series(varbal, name=variant))
            else:
                self._bal_not_found.append(variant)
        self.BAL = pd.concat(dfs, axis=1).T

    def _read_prn(self, file):
        with open(file,'r') as f:
            f = f.read().splitlines()
            data = []
            for line in f:
                line = line.split()
                data.append(line)
            units = data[0]
            headers = data[1]
            for j, line in enumerate(data):
                if len(line) == 0:
                    data = data[2:j]
                    break
        if headers[0] == 'Period' and (len(units) == len(headers)-1):
            units = [''] + [f'[{unit}]' for unit in units]
            headers = [f'{name} {unit}'.strip() for name, unit in zip(headers, units)]
        elif len(units) == len(headers):
            headers = [f'{name} [{unit}]'.strip() for name, unit in zip(headers, units)]
            print('Keine Zeitleiste')
        else:
            print('Header und Einheiten stimmen nicht überein..')

        df = pd.DataFrame({head: dat for head, dat in zip(headers,list(map(list, zip(*data))))})
        for col in df:
            df[col] = pd.to_numeric(df[col],errors='coerce')
        df = df.set_index('Period')
        df = PeriodToDatetimeIndex(df)
        return df

    def _merge_prns(self, prns):
        df = pd.concat(prns,axis=1)
        df = df.loc[:,~df.columns.duplicated()]
            
        idx = []
        for col in df.columns:
            #pd.to_numeric(df[col],errors='coerce')
            col = re.split(r'(?<!MF)_A_',col)
            id1 = ''
            id2 = ''
            if len(col) != 2:
                found = False
                for item in col:
                    r = re.search(r'\[([A-Za-z0-9_]+)\]',item)
                    if r:
                        if r[1] == 'UserEquation':
                            found = True
                            id1 = 'UserEquation'
                            id2 = item.split()[0]
                    if item.startswith('MF_'):
                        found = True
                        id1 = 'Coupling'
                        id2 = item
                    if 'amb' in item:
                        found = True
                        id1 = 'Amb'
                        id2 = item
                if found == False:
                    id2 = col[0]
                    id1 = ''
                    self._unmatched.append(f'unmatched column: {col}')
            else:
                id2 = f'{col[0]} {col[1].split()[1]}'
                id1 = col[1].split()[0]
            idx.append((id2,id1))

        if len(df.columns) == len(idx):
            df.columns = pd.MultiIndex.from_tuples(idx)
        else:
            return
        df.sort_index(axis=1,inplace=True)
        df.columns.rename(['Value', 'Airnode'],inplace=True)
        return df

    def _loadPRN(self):
        print('Lade PRN-Sheets...')
        PRN_DATA = {}
        for v, variant in enumerate(self.variants):
            files = []
            res_folder = os.path.join(self.path, variant, 'Results')
            if os.path.isdir(res_folder):
                files = [os.path.join(self.path, variant, 'Results', file) for file in os.listdir(res_folder) if file.endswith('.prn') and file.startswith('AddOut')]
                if len(files) > 0:
                    try:
                        PRN_DATA[variant] = self._merge_prns([self._read_prn(prn) for prn in files])
                        #print(f'{variant} hat geklappt')
                    except ValueError as e:
                        print(e)
                        pass
                else:
                    print(f'{variant} nicht gefunden...')
                    self._prn_not_found.append(variant)
              
            
            if self.n_variants >1:
                running_bar(v, self.n_variants)


        if len(PRN_DATA) > 0:
            while True:
                try:
                    self.PRN = pd.concat(PRN_DATA,axis=1)
                except pd.errors.InvalidIndexError:
                    for key, item in PRN_DATA.items():
                        if len(item) != 8760:
                            print(f'{key} hat nur {len(item)} Einträge und wird deshalb übersprungen')
                            del PRN_DATA[key]
                            continue
                break
        else:
            print('Keine PRN-Dateien gefunden...')

    def _getFromPRN(self, **kwargs):
        if len(kwargs) == 0:
            return self.PRN
        m = []
        for key, item in kwargs.items():
            if key in self.PRN.columns.names:
                if isinstance(item, str):
                    item = [item]
                m.append(self.PRN.columns.get_level_values(key).isin(item))
        return self.PRN.loc[:,np.array(m).T.all(axis=1)]


    def getRandom(self, by, where=None, choice=None, n=1, **kwargs):
        if where is None:
            _where = (lambda s: s==s)
        else:
            _where = where
        cols = self.VariantCollection.where(_where).dropna().groupby(by).sample(n)[by].reset_index().set_index(by).squeeze().to_dict()
        if choice in cols:
            print(f'Wähle zufällige Varainten nach {by} aus. Wähle eine Variante der Gruppe {choice}\nIch nehme: {cols[choice]}')
            return self._getFromPRN(**kwargs)[cols[choice]]
        else:
            return {self.VariantCollection.at[name, 'FensterModel']: group for name, group in self._getFromPRN(**kwargs)[list(cols.values())].groupby(level=0, axis=1)}
