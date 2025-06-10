import os, pickle
from training_data_from_torrent import ShatteredList

sl = ShatteredList(50, "/tmp/reddit/vl/fixed")

for i in range(len(os.listdir("/tmp/reddit/vl/submissions"))):
    with open(f"/tmp/reddit/vl/submissions/{i}.bin", "rb") as f:
        print(f"{i}.bin | {len(pickle.load(f))}")
        try:
            for s in pickle.load(f):
                sl.append(s)
        except EOFError:
            continue



