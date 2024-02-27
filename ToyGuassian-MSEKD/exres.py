file = open("res4.txt")
import numpy as np
# reading the file as a list line by line
Lines = file.readlines()
Res = {}
for Line in Lines:
    if Line[0] != "#":
        Line = Line.split(" ")
        if len(Line)>2:
            Para = float(Line[0])
            Acc = float(Line[1].split("(")[1].split(")")[0])
            Pdist = float(Line[2])
            if Para in Res:
                Res[Para].append([Acc, Pdist])
            else:
                Res[Para]=[[Acc, Pdist]]
for Para in Res:
    result = np.array(Res[Para])

    print(Para, np.std(result, 0))