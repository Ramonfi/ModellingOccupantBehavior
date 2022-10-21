from . import trnpy
from .toolbox import comf
from .toolbox import plot
from .toolbox import stats
from .toolbox import utils

from .database import *
from .style import *
from .templates import *

### Datenbank Import
try:
    from .database import DB, AMB, IND
except ImportError:
    database.getDB()
    from .database import  DB, AMB, IND