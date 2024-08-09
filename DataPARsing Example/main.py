   #this line makes everything else work...
#if something doesn't work, it's probably a problem in DataPARsing.py
#DataPARsing.py MUST BE PRESENT IN THE SAME DIRECTORY (folder) AS THIS FILE
from DataPARsing import * #import EVERYTHING from DataPARSing so that you can call functions directly


from random import randint

#if you want to play with neural networks, be my guest. So far they have not done much better than simply correlating variables.
from nn2 import NeuralNetwork
from nn3 import NeuralNetwork as NN3

# READ EXCEL FILE
xlfile = "HD_K1_v2.xlsx"
#getDataQuick prompts you with some questions:
#   1: do you need to pull new data from Excel? This will take a while if yes. There is already one cached data pull, so you don't need to do this on your first run.
#   2: if not, which of these numbered files is the "pickled" data from a previous pull? (The numbers correspond to dates and times)
data = getDataQuick(xlfile) # the variable "data" holds ALL of the data in a dictionary, accessible by tags
datalist = [data[key] for key in data] #not necessary, but you can reference data by datalist[n]


#EXAMPLE: Get nicer names for data by assigning names to each interesting part of the raw data
speed = data["KAL.KAL_DeltaV.DRV25219 speed"]
hd_level = data["KAL.KAL_DeltaV.LIC13610"]
speed = data["KAL.KAL_DeltaV.DRV25219 speed"]
hdk1_flow = data["KAL.KAL_DeltaV.FI212029"]

#EXAMPLE: Cleanse data
Cleanse = True #set this to False to skip the next steps...
if Cleanse:
    print("LSE...") #printing is just so you can see what's going on. It is not necessary
    low_speed_events = lse = speed.get_event_timestamps(under,1400) #get events where "speed" is "under" "1400"
    print("BFE...")
    bad_flow_events = bfe = hdk1_flow.smooth().slope().get_event_timestamps(not_within, [-150,150])
    #get events where the rate-of-change of "hdk1_flow" is outside the range, -150 to 150

    print("ET...")
    et = lse.combine(bfe,"OR") #combine the two events-objects. "OR" means if either event is happening, use it. "AND" would only look at when both events are happening.
    print("ET PAD...")
    et.pad(15) #make each event longer by 15mins in each direction (add some padding)
    #regular_level_events = rle = hd_level.get_event_timestamps(over,85)
    for ts in data: #for each object (each tag) in the data you pulled, remove those timestamps with "cleanse_timestamps"
        print(f"CLEANSING {data[ts].tag}...")
        data[ts] = data[ts].cleanse_timestamps(et)
        
#after cleansing this way, you will need to re-declare your variables from the data you cleansed
speed = data["KAL.KAL_DeltaV.DRV25219 speed"]
hd_level = data["KAL.KAL_DeltaV.LIC13610"]
speed = data["KAL.KAL_DeltaV.DRV25219 speed"]
hdk1_flow = data["KAL.KAL_DeltaV.FI212029"]


#EXAMPLE: Plot data
qplot([ #you can pass one data piece or a list of data pieces. This is in a list (hence the brackets)
    speed, 
    [hdk1_flow, hdk1_flow.smooth(50)], #a list inside the bigger list will force these onto the same plot
    #this shows a timeseries object, and a smoothed version
    [speed, low_speed_events]
    #this shows a timeseries object, and the events which we pulled earlier. If you are not cleansing, this will raise an error
    ])
