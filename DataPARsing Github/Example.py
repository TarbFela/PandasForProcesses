   #this line makes everything else work...
#if something doesn't work, it's probably a problem in DataPARsing.py
#DataPARsing.py MUST BE PRESENT IN THE SAME DIRECTORY (folder) AS THIS FILE
from DataPARsing import *

# READ EXCEL FILE
xlfile = "HD_K1_v2.xlsx"
data = getDataQuick(xlfile)
datalist = [data[key] for key in data] #not necessary, but you can reference data by datalist[n]

 #define TimeSeries-es, perform whatever calcs are relevant
speed = data["KAL.KAL_DeltaV.DRV25219 speed"]
hd_level = data["KAL.KAL_DeltaV.LIC13610"]
speed = data["KAL.KAL_DeltaV.DRV25219 speed"]
hdk1_flow = data["KAL.KAL_DeltaV.FI212029"]
weight = data["KAL.KAL_DeltaV.K1_REEL_BD"]
weather_temp = data["KAL.Envirosuite.MonW1Var3"]
weather_pres = data["KAL.Envirosuite.MonW1Var6"]



Cleanse = True
if Cleanse:#input("\nCleanse? (y/n): \n") == "y":
    low_speed_events = lse = speed.get_event_timestamps(under,1400)
    #regular_level_events = rle = hd_level.get_event_timestamps(over,85)
    for ts in data:
        for et in [lse]:
            data[ts] = data[ts].cleanse_timestamps(et)


speed = data["KAL.KAL_DeltaV.DRV25219 speed"]
hdk1_flow = data["KAL.KAL_DeltaV.FI212029"]
weight = data["KAL.KAL_DeltaV.K1_REEL_BD"]
#consistency = data["KAL.KAL_DeltaV.CIC212001"]
hd_level = data["KAL.KAL_DeltaV.LIC13610"]
flblend = data["KAL.KAL_DeltaV.FFIC_K1_FL_BLEND OCC Flow SP"]
base_bw = data["KAL.KAL_DeltaV.BASE_BW"]
coat_weight = data["KAL.CALC.K1_BarCoatWeight_Running"]

#changing_speed_events = cse = speed.slope().get_event_timestamps(not_within,[-4,4])
incrementing_speed_events = ice = speed.slope().get_event_timestamps(within,[3,20])
ice.pad(10)

HDSMOOTHFACT = 25

ds = ice.getDeltas(speed,Events,percents=True)
dl = ice.getDeltas(hd_level.smooth(HDSMOOTHFACT),Events,percents=True)
dsfb = ice.getDeltas(speed*base_bw.smooth(20)*flblend,Events,percents=True)
qplot([ds,dl,dsfb,[speed.normalize(),hdk1_flow.normalize().smooth(HDSMOOTHFACT),base_bw.normalize(),flblend.normalize()]])


def testcorr(ts):
    return ts.corr(hdk1_flow)

def add(a,b):
    return a + b * bcws2(a,b,hdk1_flow,2,False)
def mul(a,b): return a *b
def div(a,b): return a /b








