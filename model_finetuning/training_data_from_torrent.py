#Download finetuning data from Academic Torrents https://academictorrents.com/details/1614740ac8c94505e4ecb9d88be8bed7b6afddd4
#Only contains data up to December 2024
import configparser, libtorrent, zstandard, os, time, queue, json, random, sys, shutil, pickle, threading, copy
from datetime import datetime

TORRENT = "reddit_archive.torrent"

THREAD_NUM = 4

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

MIN_DATA_LEN = 10000000
MAX_DATA_LEN = 14000000
data_len = random.randint(MIN_DATA_LEN, MAX_DATA_LEN)

eval_len = int(data_len/10)

dataq = queue.Queue()

available_parts = []

def download_torrent():
    global ses
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
    #print("Thread converted")
    return out

def add_comment(comm, arr):
    global known_ids
    try:
        s = arr[known_ids[str(comm["link_id"])[3:]]]
        s["children"].append(comm)
        return True
    except:
        return False

def narrow_submissions(sub):
    global sub_range
    start = sub_range[sub][0]
    end = sub_range[sub][1]
    return vdsubs.part(start, end)

class ShatteredList:
    def __init__(self, sfl, dir):
        self.sfl = sfl
        self.lenf = 0
        self.lock = threading.RLock()
        self.len = 0
        self.loaded_num = -1
        self.dir = dir
        if os.path.exists(dir):
            self.lenf = len(os.listdir(self.dir))
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
        #sys.stdout.write(f"\rLoading index {ind}")
        #sys.stdout.flush()
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
        self.load_index(self.len-1)
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
    
    def __iter__(self):
        return SLIter(self)
    
    def part(self, start, end):
        return SLIter(self, start, end)

class SLIter:
    def __init__(self, sl, start=0, end=None):
        self.index = start
        self.sl = sl
        if end is None:
            self.end = self.sl.len
        else:
            self.end = end

    def __next__(self):
        if self.index >= self.end:
            raise StopIteration
        try:
            obj = self.sl[self.index]
        except (IndexError, FileNotFoundError): 
            raise StopIteration
        self.index += 1
        return obj
    
    def __iter__(self):
        return self
#data generation starts here.
print("Generating training and eval data")

training_data = ""
eval_data = ""

known_ids = {}

sub_range = {}

sub_comms = {}

def data_gen():
    global vdsubs, available_parts, sub_comms
    if not os.path.exists("/tmp/reddit/vl"):
        os.mkdir("/tmp/reddit/vl")
        vdsubs = ShatteredList(50, "/tmp/reddit/vl/submissions")
        
        for sub in subreddits:
            print(f"Loading valid submissions data for subreddit r/{sub}")
            sub_range[sub] = []
            
            subnum = 0
            with open(f"/tmp/reddit/unzipped/{sub}_submissions.ndjson", "rb") as f:
                print()
                sub_range[sub].append(len(vdsubs))
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
                            known_ids[s["id"]] = len(vdsubs)
                            sub_comms[s["id"]] = 0
                            s["children"] = []
                            vdsubs.append(s)
                            subnum += 1
                            sys.stdout.write(f"\rAdded {subnum} submissions for subreddit r/{sub}")
                            sys.stdout.flush()
                sub_range[sub].append(len(vdsubs))
                print()

            print(f"Loading valid comments data for subreddit r/{sub}")
            commnum = 0
            with open(f"/tmp/reddit/unzipped/{sub}_comments.ndjson", "rb") as f:
                comm_limit = subnum * int(mincomm)
                print()
                for line in f:
                    s = json.loads(line.decode("utf-8"))
                    badcomm = False
                    if start_time_unix < int(s["created_utc"]) and end_time_unix > int(s["created_utc"]) and str(s["link_id"])[3:] in known_ids:
                        for b in badwords:
                            if b in s["body"]:
                                badcomm = True
                        if badcomm or sub_comms[str(s["link_id"][3:])]>mincomm:
                            continue
                        else:
                            #subiter = narrow_submissions(sub)
                            if add_comment(s, vdsubs):
                                commnum += 1
                                sub_comms[str(s["link_id"][3:])] += 1
                                known_ids[s["id"]] = known_ids[str(s["link_id"])[3:]]
                                sys.stdout.write(f"\rAdded {commnum} comments for subreddit r/{sub} (Limit: {comm_limit})")
                                sys.stdout.flush()
                        if commnum > comm_limit:
                            break
                print()
    else:
        vdsubs = ShatteredList(50, "/tmp/reddit/vl/submissions")
        #i=0
        #prevsub = None
        #for s in vdsubs:
        #    if s["subreddit"] not in sub_range:
        #        sub_range[s["subreddit"]] = [i]
        #        sub_range[prevsub].append(i)    
        #    i += 1
        #    prevsub = s["subreddit"]
        print("Previous data has been detected, and that data will be used.")
        print("If you do not want to reuse the data from the last time this script was run, delete /tmp/reddit before running.")
    available_parts = list(range(vdsubs.lenf))


def still_running():
    return len(training_data) <= data_len or len(eval_data) <= eval_len

def org_comments_thread(tnum):
    global training_data, dataq, vdsubs, available_parts
    #print(f"Comment organization thread {tnum} starting")
    subs = []
    while still_running() and len(available_parts)>0:
        valid = True
        while len(subs)<1:
            #print(f"thread-{tnum}: obtaining lock")
            vdsubs.lock.acquire()
            #print(f"thread-{tnum}: lock obtained")
            if len(available_parts)<1:
                #print("There was not enough training data. Please close script with ctrl-c and change config to add more data (lower min comments or increase date range)")
                return
            part = available_parts.pop(random.randrange(len(available_parts)))
            vdsubs.load_index(vdsubs.sfl*part)
            subs = copy.copy(vdsubs.loaded)
            #print(f"thread-{tnum}: submissions obtained")
            vdsubs.lock.release()
        sub = subs.pop(random.randrange(len(subs)))
        #print(f"thread-{tnum}: sub selected")
        iscomm = random.choice([True, True, False])
        if iscomm and len(sub["children"])>0:
            #print(f"thread-{tnum}: sub has comments")
            comms = sub["children"]
            sub["children"] = []
            finalcomm = comms.pop(random.randrange(len(comms)))
            replies = [finalcomm]
            post = sub
            while not replies[-1]["parent_id"]==f"t3_{str(post['id'])}":
                for c in comms:
                    if str(replies[-1]["parent_id"])[3:]==c["id"]:
                        replies.append(c)
                        break
                else:
                    valid=False
                    break
            if not valid:
                continue
            else:
                replies.reverse()
                dataq.put(convert_thread(post, replies))
        else:
            dataq.put(convert_post(sub))

def add_data_thread():
    global training_data, eval_data
    while still_running():
        d = dataq.get()
        if len(training_data)<data_len:
            training_data += d
            sys.stdout.write(f"\rTraining data completion: {len(training_data)/data_len}")
            sys.stdout.flush()
            if not len(training_data)<data_len:
                print()
        else:
            eval_data += d
            sys.stdout.write(f"\rEval data completion: {len(eval_data)/eval_len}")
            sys.stdout.flush()

if __name__=="__main__":
    download_torrent()
    data_gen()
    
    threads = []

    for i in range(THREAD_NUM):
        threads.append(threading.Thread(target=org_comments_thread, args=(i,)))
        threads[-1].start()

    add_data_thread()

    with open("training_data.txt", "w") as f:
        f.write(training_data)

    with open("eval_data.txt", "w") as f:
        f.write(eval_data)