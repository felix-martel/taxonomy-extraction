import re
from .uri import PREFIXES

URL_PATTERN = re.compile("^<(?P<prefix>https?://.+/(.*#)?)(?P<uri>.*)>$")
STR_PATTERN = re.compile('^"(?P<value>.*?)"@(?P<lang>.*)$')
RAW_STR_PATTERN = re.compile('^"(.*)"$')
VAR_PATTERN = re.compile('^"(?P<value>.*?)"\^\^(?P<formt>.*)$')

DIR = "data/dbpedia/"

files = [
    DIR + 'dbpedia_2016-10.nt', 
    #DIR + 'out_degree_en.ttl', 
    DIR + 'mappingbased_objects_en.ttl', 
    DIR + 'labels_en.ttl', 
    DIR + 'instance_types_transitive_en.ttl', 
    DIR + 'mappingbased_literals_en.ttl', 
    DIR + 'persondata_en.ttl', 
    DIR + 'instance_types_en.ttl', 
    DIR + 'pnd_en.ttl',
]

def split_line(l):
    a = l.split(" ")
    h = a[0]
    r = a[1]
    t = " ".join(a[2:-1])
    return h, r, t
    
def get_identifier(s, shorten=False):
    if re.search(RAW_STR_PATTERN, s):
        return "<STRING>"
    m = re.search(STR_PATTERN, s)
    if m:
        return "<LABEL:" + m.group("lang") + ">"
    m = re.search(VAR_PATTERN, s)
    if m:
        return m.group("formt")
    if shorten:
        m = re.search(URL_PATTERN, s)
        if m:
            pref = m.group("prefix")
            if pref in PREFIXES.longs:
                return PREFIXES.to_short(pref) + ":" + m.group("uri")
    return s

def iter_files(files):
    for file in files:
        with open(file, "r") as f:
            for line in f:
                h, r, t = split_line(line)
                yield h, r, t, file
                
                
                