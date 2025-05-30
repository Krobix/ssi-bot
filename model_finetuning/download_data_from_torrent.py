#Download finetuning data from Academic Torrents https://academictorrents.com/details/1614740ac8c94505e4ecb9d88be8bed7b6afddd4
#Only contains data up to December 2024
import configparser, libtorrent, zstandard, os, time, queue, json, random, sys, shutil, pickle, threading
from datetime import datetime

TORRENT = "reddit_archive.torrent"

THREAD_NUM = 8

ti = libtorrent.torrent_info(TORRENT)
ses = libtorrent.session()
ses.listen_on(6881, 6891)

config = configparser.ConfigParser()
config.read("dataset.ini")

conf = config['DEFAULT']

subreddits = conf['training_subreddits'].split(",")

downloading_file_indexes = []
priorities = []

if conf['min_comments']:
    mincomm = int(conf['min_comments'])
else:
    print("Set a minimum number of comments for this script to work properly, otherwise way too much data will be used")
    sys.exit()

start_time_unix = int(time.mktime(datetime.fromisoformat(conf['start_date']).timetuple()))
end_time_unix = int(time.mktime(datetime.fromisoformat(conf['end_date']).timetuple()))

badwords = conf["negative_keywords"].split(",")

MIN_DATA_LEN = 12000000
MAX_DATA_LEN = 15000000
data_len = random.randint(MIN_DATA_LEN, MAX_DATA_LEN)

eval_len = int(data_len/10)

dataq = queue.Queue()

for sub in subreddits:
    print(f"Downloading data for {sub}")
    ind = 0
    for f in ti.files():
        if (f.path.split("/")[-1]==f"{sub}_submissions.zst" or f.path.split("/")[-1]==f"{sub}_comments.zst") and (not os.path.exists(f"/tmp/{f.path}")):
            print(f"Identified {f.path} at index {ind}")
            downloading_file_indexes.append(ind)
        ind += 1

for i in range(ti.num_files()):
    if(i in downloading_file_indexes):
        priorities.append(5)
    else:
        priorities.append(0)

handle = ses.add_torrent(ti, "/tmp")
handle.prioritize_files(priorities)

print("Starting download")
while (not handle.status().is_finished):
    print(f"Download progress: {handle.status().progress}")
    time.sleep(1)
ses.remove_torrent(handle)
del ses

for sub in subreddits:
    print(f"Decompressing submissions and comments for r/{sub}")
    CS = 10000000

    if os.path.exists(f"/tmp/reddit/unzipped/{sub}_submissions.ndjson"):
        continue

    if not os.path.exists("/tmp/reddit/unzipped"):
        os.mkdir("/tmp/reddit/unzipped")

    with open(f"/tmp/reddit/subreddits24/{sub}_submissions.zst", "rb") as f:
        data = f.read()
    reader = zstandard.ZstdDecompressor().stream_reader(data)
    datalen = len(data)
    tot_read = 0
    with open(f"/tmp/reddit/unzipped/{sub}_submissions.ndjson", "ab+") as f:
        while True:
            chunk = reader.read(CS)
            tot_read += CS
            #print(f"Read {int(tot_read/CS)} chunks")
            if not chunk:
                break
            f.write(chunk)

    with open(f"/tmp/reddit/subreddits24/{sub}_comments.zst", "rb") as f:
        data = f.read()
    reader = zstandard.ZstdDecompressor().stream_reader(data)
    datalen = len(data)
    tot_read = 0
    with open(f"/tmp/reddit/unzipped/{sub}_comments.ndjson", "ab+") as f:
        while True:
            chunk = reader.read(CS)
            tot_read += CS
            #print(f"Read {int(tot_read/CS)} chunks")
            if not chunk:
                break
            f.write(chunk)

    print("Decompression finished; Generating training data")

#trying to use the db to try and generate data was really annoying so im just doing it myself
def convert_post(p, end_tag=True):
    out = ""
    
    if p['is_self']:
        out += f"<|soss r/{p['subreddit']}|>"
    else:
        out += f"<|sols r/{p['subreddit']}|>"

    out += f"<|sot|>{p['title']}"

    if p['is_self']:
        out += f"<|sost|>{p['selftext']}"
    if end_tag:
        if p['is_self']:
            out += "<|eoss|><|endoftext|>\n"
        else:
            out += "<|eols|><|endoftext|>\n"
    return out


def convert_thread(post, replies):
    out = convert_post(post, end_tag=False)
    if out is None:
        return None
    
    parent_author = None
    parent_2_author = None
    for r in replies:
        if r['author']==post['author']:
            out += "<|soopr|>"
        elif r['author'] == parent_2_author:
            out += "<|soocr|>"
        else:
            out += f"<|sor u/{r['author']}|>"
        out += r['body']
        parent_2_author = parent_author
        parent_author = r["author"]

    if post['is_self']:
        out += "<|eoss|><|endoftext|>\n"
    else:
        out += "<|eols|><|endoftext|>\n"

    return out

class ShatteredList:
    def __init__(self, sfl, dir):
        self.sfl = sfl
        self.lenf = 0
        self.lock = threading.RLock()
        self.len = 0
        self.loaded_num = -1
        self.dir = dir
        if os.path.exists(dir):
            self.lenf = len([name for name in os.listdir(self.dir) if os.path.isfile(name)])
            for fn in os.listdir(self.dir):
                with open(f"{self.dir}/{fn}", "rb") as f:
                    self.len += len(pickle.load(f))
            self.load_index(0)
        else:
            os.mkdir(self.dir)
            self.loaded = []
            self.loaded_num = 0

    def path(self, num):
        return f"{self.dir}/{num}.bin"
    
    def commit(self):
        self.lock.acquire()
        if self.loaded_num<0:
            self.lock.release()
            return
        with open(self.path(self.loaded_num), "wb") as f:
            pickle.dump(self.loaded, f)
        self.lock.release()

    def load_index(self, ind):
        #returns "real index"
        self.lock.acquire()
        loc = int(ind / self.sfl)
        if loc == self.loaded_num:
            self.lock.release()
            return ind % self.sfl
        self.commit()
        self.loaded_num = loc
        with open(self.path(self.loaded_num), "rb") as f:
            self.loaded = pickle.load(f)
        #assert (len(self.loaded)==self.sfl) or (self.loaded_num==self.lenf), f"List length wrong T-T len(self.loaded)={len(self.loaded)}"
        self.lock.release()
        return ind % self.sfl
    
    def append(self, obj):
        self.lock.acquire()
        if len(self.loaded) >= self.sfl:
            self.commit()
            self.lenf += 1
            self.loaded_num += 1
            self.loaded = []
        self.loaded.append(obj)
        self.len+=1
        self.lock.release()

    def __getitem__(self, ind):
        self.lock.acquire()
        nind = self.load_index(ind)
        self.lock.release()
        return self.loaded[nind]
    
    def pop(self, ind):
        self.lock.acquire()
        nind = self.load_index(ind)
        obj = self.loaded.pop(nind)
        while self.loaded_num < self.lenf:
            self.load_index((self.loaded_num+1)*self.sfl)
            n = self.loaded.pop(0)
            if len(self.loaded)==0:
                os.remove(self.path(self.loaded_num))
                self.lenf -= 1
            self.load_index((self.loaded_num-1)*self.sfl)
            self.loaded.append(n)
            self.load_index((self.loaded_num+1)*self.sfl)
        self.len -= 1
        self.load_index(ind)
        self.lock.release()
        return obj
    
    def __len__(self):
        return self.len


#data generation starts here.
print("Generating training and eval data")

training_data = ""
eval_data = ""

vdsubs = ShatteredList(10000, "/tmp/reddit/vl/submissions")
vdcomms = ShatteredList(10000, "/tmp/reddit/vl/comments")

known_ids = []


if not os.path.exists("/tmp/reddit/vl"):
    os.mkdir("/tmp/reddit/vl")
    
    for sub in subreddits:
        print(f"Loading valid submissions data for subreddit r/{sub}")
        
        subnum = 0
        with open(f"/tmp/reddit/unzipped/{sub}_submissions.ndjson", "rb") as f:
            print()
            for line in f:
                badsub = False
                s = json.loads(line.decode("utf-8"))
                if s["num_comments"] >= mincomm and start_time_unix < int(s["created_utc"]) and end_time_unix > int(s["created_utc"]):
                    if "selftext" in s:
                        for b in badwords:
                            if b in s["selftext"] or b in s["title"]:
                                badsub = True
                    if badsub:
                        continue
                    else:
                        known_ids.append(s["id"])
                        vdsubs.append(s)
                        subnum += 1
                        sys.stdout.write(f"\rAdded {subnum} submissions for subreddit r/{sub}")
                        sys.stdout.flush()
            print()

        print(f"Loading valid comments data for subreddit r/{sub}")
        commnum = 0
        with open(f"/tmp/reddit/unzipped/{sub}_comments.ndjson", "rb") as f:
            comm_limit = subnum * int(mincomm*2)
            print()
            for line in f:
                s = json.loads(line.decode("utf-8"))
                badcomm = False
                if start_time_unix < int(s["created_utc"]) and end_time_unix > int(s["created_utc"]) and str(s["parent_id"])[3:] in known_ids:
                    for b in badwords:
                        if b in s["body"]:
                            badcomm = True
                    if badcomm:
                        continue
                    else:
                        known_ids.append(s["id"])
                        vdcomms.append(s)
                        commnum += 1
                        sys.stdout.write(f"\rAdded {commnum} comments for subreddit r/{sub} (Limit: {comm_limit})")
                        sys.stdout.flush()
                    if commnum > comm_limit:
                        break
            print()
else:
    print("Previous data has been detected, and that data will be used.")
    print("If you do not want to reuse the data from the last time this script was run, delete /tmp/reddit before running.")

if os.path.exists("/tmp/reddit/thr"):
    shutil.rmtree("/tmp/reddit/thr")
os.mkdir("/tmp/reddit/thr")


def add_training_data(tnum):
    #os.mkdir(f"/tmp/reddit/thr/{tnum}")
    vdcomms.lock.acquire()
    print(f"Preparing data for thread {tnum}...")
    subs = vdsubs
    comms = ShatteredList(5000, f"/tmp/reddit/thr/{tnum}")
    mylen = int(len(vdcomms)/THREAD_NUM)
    start, end = mylen*tnum, mylen*(tnum+1)
    for i in range(start, end):
        try:
            comms.append(vdcomms[i])
        except IndexError:
            break
    print(f"Finished loading data for thread {tnum}")
    vdcomms.lock.release()
    
    while len(training_data) < data_len or len(eval_data) < eval_len:
        newdat = ""
        comment = random.choice([True, True, False])
        if not comment:
            s = subs.pop(random.randrange(len(subs)))
            newdat = convert_post(s)
        else:
            s = comms.pop(random.randrange(len(comms)))
            replies = [s]
            post = None
            while post is None:
                if replies[-1]["parent_id"].startswith("t3"):
                    for s2 in subs:
                        if s2["id"] == str(replies[-1]["parent_id"])[3:]:
                            post = s2
                            break
                    else:
                        break
                else:
                    for s2 in comms:
                        if s2["id"] == str(replies[-1]["parent_id"])[3:]:
                            replies.append(s2)
                            break
                    else:
                        break

            if post is None:
                continue

            replies.reverse()
            newdat = convert_thread(post, replies)   

        dataq.put(newdat)    

def add_to_data():
    global eval_data, training_data
    while len(training_data) < data_len or len(eval_data) < eval_len:
        newdat = dataq.get()
        if len(training_data) >= data_len:
            eval_data += newdat
            print(f"Eval data progress: {len(eval_data) / eval_len}")
        else:
            training_data += newdat
            print(f"Training data progress: {len(training_data) / data_len}")

threads = []
for n in range(THREAD_NUM):
    threads.append(threading.Thread(target=add_training_data, args=(n,)))
    threads[n].start()

add_to_data()

with open("training_data.txt", "w") as f:
    f.write(training_data)

with open("eval_data.txt", "w") as f:
    f.write(eval_data)