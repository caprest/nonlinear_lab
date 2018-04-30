import string
import datetime
import pandas as pd
import numpy as np
import os
import re


def path_generator(year, month, place, amedas_root):
    AtoZ = string.ascii_uppercase[0:27]
    AtoZ = str(AtoZ)
    zeroto_z = str(0) + str(123456789) + AtoZ
    fmt = "{year}_{1or2}/AM10{year_2:02d}{month_2:02d}/A{year_1}{month_1}{place:05d}.CSV"
    param = {
        "year": year,
        "year_2": year % 100,
        "1or2": "1" if month <= 6 else "2",
        "month_2": month,
        "year_1": zeroto_z[year - 1990],
        "month_1": zeroto_z[month],
        "place": place
    }
    return os.path.join(amedas_root, fmt.format(**param))


def month_converter(year, month, place, amedas_root):
    DATAPATH = path_generator(year, month, place, amedas_root)
    with open(DATAPATH, "r") as f:
        data = f.readlines()
    df = pd.DataFrame([],
                      columns=["datetime", "precipitation", "temparature", "wind_direction", "wind_speed",
                               "sunshine", ])
    for i in range(len(data) // 145):
        idx = data[145 * i]
        if idx.strip().split(",")[1] == '4':
            raw = [datum.strip().split(",")[1:6] for datum in data[145 * i + 1:145 * i + 145]]
            temp_df = pd.DataFrame(raw,
                                   columns=["precipitation", "temparature", "wind_direction", "wind_speed", "sunshine"])
            param = {"year": year, "month": month, "day": i + 1}
            temp_df["datetime"] = pd.date_range("{year}-{month:02d}-{day:02d} 00:00".format(**param),
                                                "{year}-{month:02d}-{day:02d} 23:50".format(**param),
                                                freq="10T") + datetime.timedelta(minutes=10)
            df = pd.concat([df, temp_df])
        else:
            print("Formatting Error, datatype = {}".format(idx.strip().split(",")[1]))
            raise
    return df

def line_parser(line):
    idxpattern = re.compile(
        r"^(?P<code>[0-9]{5})(?P<name>[ぁ-んァ-ン一-龥ヶ]+)　*(?P<kana>[^\s]+)\s*(?P<abb>[^\s]+)\s*(?P<cor>[0-9]{10,11})\s*(?P<alt>[\-0-9]+)")
    m = idxpattern.match(line)
    cor = m.group("cor")
    latitude = cor[0:5]
    longitude = cor[5:11]
    return [m.group("code"), m.group("name"), m.group("kana"), m.group("abb"), latitude, longitude, m.group("alt")]


def get_place_code(year, month, cityname, amedasdir):
    fmt = "{year}_{1or2}/"
    param = {
        "year": year,
        "1or2": "1" if month <= 6 else "2",
    }
    dirname = os.path.join(amedasdir, fmt.format(**param), "idx")
    path1 = os.path.join(dirname, "idx{year}.{month:02d}".format(year=year, month=month))
    path2 = os.path.join(dirname, "sidx{year}.{month:02d}".format(year=year, month=month))
    found_flag = 0
    for path in [path1, path2]:
        with open(path, encoding="cp932") as f:
            data = f.readlines()
        for datum in data:
            idx = datum.find(cityname)
            if idx != -1:
                found_flag = 1
                break
        if found_flag == 1:
            break
        print("not found in {}".format(path))
    if found_flag == 0:
        print("{year}_{month}_{cityname} not found".format(year=year, month=month, cityname=cityname))
        ret = -1
    else:
        parsed_data = []
        for datum in data:
            try:
                datum = line_parser(datum)
                parsed_data.append(datum)
            except AttributeError:
                print("WARNING:Parse error in the following line")
                print(datum)
        df = pd.DataFrame(parsed_data, columns=["code", "city", "kana", "abb", "latitude", "longitude", "altitude"])
        ret = df.loc[df["city"] == cityname, "code"]
        if len(ret) > 0:
            ret = ret.iloc[0]
        else:
            ret = -1

    return int(ret)
