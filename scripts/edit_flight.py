'''
Edits X coordinate by "value" of parsed flight log to adjust for inaccuracies when plotting the flight path
'''

import argparse
import csv

parser = argparse.ArgumentParser(description="Preprocess folder from oi_download_dataset")
parser.add_argument("--input", "-i", dest="FILE")
parser.add_argument("--value", "-v", dest="VALUE")
args = parser.parse_args()

file = args.FILE
with open(file,"r") as source:
    rdr = csv.reader(source)
    next(rdr) 
    with open(file + "2","w+") as result:
        wtr= csv.writer(result)
        wtr.writerow(("X", "Y", "Z"))
        
        for r in rdr:
            xvalue = float(r[0]) - float(args.VALUE)
            wtr.writerow((xvalue, r[1], r[2]))