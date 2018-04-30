import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import re
import os
import gzip
import shutil
import subprocess
import sys
import csv
from tqdm import tqdm

import pandas as pd
import re
import os


REPORTDIR = '/home/meip-users/climate_economy/data/reuters/report'
OUTDIR = "/home/meip-users/climate_economy/data/reuters/"
GZIPDIR = "/media/meip-users/BUFFALO/ロイター経済データ/"

name_list = os.listdir(REPORTDIR)
pattern = re.compile(r"(shintaro|kentaro).*report.*")
idx_idx_list = [name for name in name_list  if pattern.match(name)]
targetric = ["JCRc1","JASc1","CLc1","HOc1","NGc1","JPY="]
columns = ["filename"]+targetric
index_list = pd.DataFrame([],columns = columns)

def matches_year(name):
    pattern = re.compile(r".*(2006|2007|2008).*")
    m = pattern.match(name)
    ret = 1 if m else 0
    return ret
index_list["filename"] = idx_idx_list

def exists_target(target,filename):
    df = pd.read_csv(os.path.join(REPORTDIR,filename))
    selected_df = df[(df["#RIC"] == target) & (df["Status"] == "Active") & (df["Start (GMT)"].apply(matches_year)) & (df["End (GMT)"].apply(matches_year))]
    ret = 1 if len(selected_df)> 0 else 0
    return ret

def generate_target_list(targetric =["JCRc1","JASc1","CLc1","HOc1","NGc1","JPY="] ):
    name_list = os.listdir(REPORTDIR)
    pattern = re.compile(r"(shintaro|kentaro).*report.*")
    idx_idx_list = [name for name in name_list if pattern.match(name)]
    columns = ["filename"] + targetric
    index_list = pd.DataFrame([], columns=columns)
    index_list["filename"] = idx_idx_list
    for target in targetric:
        index_list[target] = index_list["filename"].apply(
            lambda filename : exists_target(target,filename)
        )
    target_list = index_list.assign(
        get = lambda df : df[targetric].apply( lambda x: x.any() , axis = 1)
    ).pipe(lambda df: df[df["get"] ==1]).assign(
        idx_list = lambda df: df.apply(lambda x: [x.index[i] for i,v in enumerate(x[0:-1]) if v == 1 ] , axis = 1),
        is_future = lambda df:df["filename"].apply(lambda x: 1 if re.match(r".*FUTURE.*",x) else 0),
    )
    return target_list


def get_filelist(v):
    filename = v["filename"]
    file_pattern = re.compile("-".join(filename.split("-")[0:-1] + ["part[0-9]{3}.csv.gz"]))
    dirname = GZIPDIR
    target_dir = os.path.join(dirname, 'FUTURES') if v["is_future"] else os.path.join(dirname, "FX_INDICES2")
    files = []
    for file in os.listdir(target_dir):
        if file_pattern.match(file):
            files.append(file)
    if v["is_future"]:
        gzipdir = os.path.join(dirname, "FUTURES")
    else:
        gzipdir = os.path.join(dirname, "FX_INDICES2")
    return files, gzipdir

def get_real_filelist(v):
    pattern = "-".join(v["filename"].split("-")[0:-1])+"-part[0-9]{3}.csv.gz"
    return sorted([filename for filename in os.listdir(OUTDIR) if re.match(pattern, filename)])


def ungzipfiles(files, gzipdir="/media/meip-users/BUFFALO/ロイター経済データ/FUTURES",
                outdir="/home/meip-users/climate_economy/data"):
    for filename in tqdm(files):
        new_filename = ".".join(filename.split(".")[0:-1])
        with gzip.open(os.path.join(gzipdir, filename), 'rb') as f_in, open(os.path.join(outdir, new_filename),
                                                                            'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)


def copyfiles(files, gzipdir, out_dir=OUTDIR):
    for filename in tqdm(files):
        read_path = os.path.join(gzipdir, filename)
        write_path = os.path.join(out_dir, filename)
        shutil.copyfile(read_path, write_path)


def copy_target_gzip(targetric = ["JCRc1","JASc1","CLc1","HOc1","NGc1","JPY="]):
    target_list = generate_target_list(targetric)
    target_list.to_csv("target_list.csv",index =None)
    for _,v in tqdm(target_list.iterrows()):
        files, gzipdir = get_filelist(v)
        copyfiles(files,gzipdir)

