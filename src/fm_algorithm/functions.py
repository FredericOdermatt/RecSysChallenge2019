import math
import pandas as pd
import numpy as np
import re
from pathlib import Path
from joblib import Parallel, delayed
import multiprocessing


GR_COLS = ["user_id", "session_id", "timestamp", "step"]

propkeys = ['1 Star', '2 Star', '3 Star', '4 Star', '5 Star', 'Accessible Hotel', 'Accessible Parking', 'Adults Only', 'Air Conditioning', 'Airport Hotel', 'Airport Shuttle', 'All Inclusive (Upon Inquiry)', 'Balcony', 'Bathtub', 'Beach', 'Beach Bar', 'Beauty Salon', 'Bed & Breakfast', 'Bike Rental', 'Boat Rental', 'Body Treatments', 'Boutique Hotel', 'Bowling', 'Bungalows', 'Business Centre', 'Business Hotel', 'Cable TV', 'Camping Site', 'Car Park', 'Casa Rural (ES)', 'Casino (Hotel)', 'Central Heating', 'Childcare', 'Club Hotel', 'Computer with Internet', 'Concierge', 'Conference Rooms', 'Convenience Store', 'Convention Hotel', 'Cosmetic Mirror', 'Cot', 'Country Hotel', 'Deck Chairs', 'Design Hotel', 'Desk', 'Direct beach access', 'Diving', 'Doctor On-Site', 'Eco-Friendly hotel', 'Electric Kettle', 'Excellent Rating', 'Express Check-In / Check-Out', 'Family Friendly', 'Fan', 'Farmstay', 'Fitness', 'Flatscreen TV', 'Free WiFi (Combined)', 'Free WiFi (Public Areas)', 'Free WiFi (Rooms)', 'Fridge', 'From 2 Stars', 'From 3 Stars', 'From 4 Stars', 'Gay-friendly', 'Golf Course', 'Good Rating', 'Guest House', 'Gym', 'Hairdresser', 'Hairdryer', 'Halal Food', 'Hammam', 'Health Retreat', 'Hiking Trail', 'Honeymoon', 'Horse Riding', 'Hostal (ES)', 'Hostel', 'Hot Stone Massage', 'Hotel', 'Hotel Bar', 'House / Apartment', 'Hydrotherapy', 'Hypoallergenic Bedding', 'Hypoallergenic Rooms', 'Ironing Board', 'Jacuzzi (Hotel)', "Kids' Club", 'Kosher Food', 'Large Groups', 'Laundry Service', 'Lift', 'Luxury Hotel', 'Massage', 'Microwave', 'Minigolf', 'Motel', 'Nightclub', 'Non-Smoking Rooms', 'On-Site Boutique Shopping', 'Openable Windows', 'Organised Activities', 'Pet Friendly', 'Playground', 'Pool Table', 'Porter', 'Pousada (BR)', 'Radio', 'Reception (24/7)', 'Resort', 'Restaurant', 'Romantic', 'Room Service', 'Room Service (24/7)', 'Safe (Hotel)', 'Safe (Rooms)', 'Sailing', 'Satellite TV', 'Satisfactory Rating', 'Sauna', 'Self Catering', 'Senior Travellers', 'Serviced Apartment', 'Shooting Sports', 'Shower', 'Singles', 'Sitting Area (Rooms)', 'Ski Resort', 'Skiing', 'Solarium', 'Spa (Wellness Facility)', 'Spa Hotel', 'Steam Room', 'Sun Umbrellas', 'Surfing', 'Swimming Pool (Bar)', 'Swimming Pool (Combined Filter)', 'Swimming Pool (Indoor)', 'Swimming Pool (Outdoor)', 'Szep Kartya', 'Table Tennis', 'Telephone', 'Teleprinter', 'Television', 'Tennis Court', 'Tennis Court (Indoor)', 'Terrace (Hotel)', 'Theme Hotel', 'Towels', 'Very Good Rating', 'Volleyball', 'Washing Machine', 'Water Slide', 'Wheelchair Accessible', 'WiFi (Public Areas)', 'WiFi (Rooms)']


def get_submission_target(df):
    """Identify target rows with missing click outs."""

    mask = df["reference"].isnull() & (df["action_type"] == "clickout item")
    df_out = df[mask]

    return df_out


def get_popularity(df):
    """Get number of clicks that each item received in the df."""

    mask = df["action_type"] == "clickout item"
    df_clicks = df[mask]
    df_item_clicks = (
        df_clicks
        .groupby("reference")
        .size()
        .reset_index(name="n_clicks")
        .transform(lambda x: x.astype(int))
    )

    return df_item_clicks


def string_to_array(s):
    """Convert pipe separated string to array."""

    if isinstance(s, str):
        out = s.split("|")
    elif math.isnan(s):
        out = []
    else:
        raise ValueError("Value must be either string of nan")
    return out


def explode(df_in, col_expl):
    """Explode column col_expl of array type into multiple rows."""

    df = df_in.copy()
    # df.loc[:, col_expl] = df[col_expl].apply(string_to_array)

    df_out = pd.DataFrame(
        {col: np.repeat(df[col].values,
                        df[col_expl].str.len())
         for col in df.columns.drop(col_expl)}
    )

    df_out.loc[:, col_expl] = np.concatenate(df[col_expl].values)
    df_out.loc[:, col_expl] = df_out[col_expl].apply(int)

    return df_out


def group_concat(df, gr_cols, col_concat):
    """Concatenate multiple rows into one."""

    df_out = df.sort_values(['predict'],ascending=False)
    df_out = df_out.drop('predict',1)
    df_out = df_out.astype({col_concat:str})
    df_out = df_out.groupby(gr_cols)[col_concat].apply(lambda x: ' '.join(x)).to_frame().reset_index()

    return df_out


def calc_recommendation(df_expl, df_pop):
    """Calculate recommendations based on popularity of items.

    The final data frame will have an impression list sorted according to the number of clicks per item in a reference data frame.

    :param df_expl: Data frame with exploded impression list
    :param df_pop: Data frame with items and number of clicks
    :return: Data frame with sorted impression list according to popularity in df_pop
    """

    df_expl_clicks = (
        df_expl[GR_COLS + ["impressions"]]
        .merge(df_pop,
               left_on="impressions",
               right_on="reference",
               how="left")
    )

    df_out = (
        df_expl_clicks
        .assign(impressions=lambda x: x["impressions"].apply(str))
        .sort_values(GR_COLS + ["n_clicks"],
                     ascending=[True, True, True, True, False])
    )

    df_out = group_concat(df_out, GR_COLS, "impressions")
    df_out.rename(columns={'impressions': 'item_recommendations'}, inplace=True)

    return df_out


def regextostr(textin):
    strlist = re.findall(r'[^\|]+',textin)

    return strlist


def getlist(listin):
    for i,textin in enumerate(listin): listin[i] = regextostr(textin)

    return listin


def onehotprop(df):
    list = df.values.tolist()
    list = getlist(list)
    propkeys_ = propkeys

    ncols = len(keys)
    keys = dict(zip(propkeys_,range(ncols)))

    # df_onehot = pd.get_dummies(pd.Series(list).apply(pd.Series).stack()).sum(level=0) # super-slow
    df_onehot = np.zeros((len(list),ncols),dtype=np.uint8)
    for i, props in enumerate(list):
        for prop in props:
            df_onehot[i,keys[prop]] = 1

    df_onehot = pd.DataFrame(df_onehot,columns=keys)

    return df_onehot, keys


def onehotid(df):
    df = df.values.tolist()
    names = list(dict.fromkeys(df))
    ncols = len(names)
    names = dict(zip(names,range(ncols)))

    df_onehot = np.zeros((len(df),ncols),dtype=np.uint8)
    for i, ids in enumerate(df):
        df_onehot[i,names[ids]] = 1

    df_onehot = pd.DataFrame(df_onehot,columns=names)

    return df_onehot, names


def onehotsession(dfin,cols):
    listkeys = []
    for col in cols:
        print(f"Encoding {col} ...")
        df, keys = onehotid(dfin[col])
        dfin = dfin.drop(col,1)
        dfin = pd.concat([dfin.reset_index(drop=True),df.reset_index(drop=True)],axis=1)
        listkeys.append(keys)

    return dfin, listkeys


def getfieldmap(listkeys):
    feat = []
    field = []
    for f,fieldkeys in enumerate(listkeys):
        for colname,idx in fieldkeys.items():
            feat.append(colname)
            field.append(f)
    fieldmap = dict(zip(feat,field))

    return fieldmap


def divider(ncore,nrows):
    divider_ = []
    for val in range(ncore):
        divider_.append(val*int((nrows/ncore)))
    divider_.append(nrows)

    return divider_


def fittarget(df,listkeys,onehotcols):
    ncols = 0
    colname = []
    propkeys_ = propkeys
    for keys in listkeys:
        ncols += len(keys)
        colname += keys.keys()
    colname = dict(zip(colname,range(ncols)))

    ncore = multiprocessing.cpu_count()
    print("Using %d CPUs ..." % ncore)

    divider_ = divider(ncore,df.shape[0])

    def vecfit(p,start,end,df,ncols,colname,onehotcols):
        df_vec = np.zeros((end-start,ncols),dtype=np.uint8)
        for i in range(start,end):
            if i%100000 == 0 : print("Processing %d th row of %d th process ..." % (i,p))
            row = df.iloc[i].to_dict()
            df_row = np.zeros(ncols,dtype=np.uint8)
            for cols in onehotcols:
                if row[cols] in colname:
                    df_row[colname[row[cols]]] = 1
            for prop in propkeys_:
                df_row[colname[prop]] = row[prop]

            df_vec[i-start] = df_row

        return p, df_vec

    vecresult = Parallel(n_jobs=ncore)(delayed(vecfit)(p,divider_[p],divider_[p+1],df,ncols,colname,onehotcols) for p in range(ncore))
    vecresult.sort()

    df_target = []
    for tup in vecresult:
        df_target.append(tup[1])

    df_target = np.concatenate(df_target,axis=0)
    df_target = pd.DataFrame(df_target,columns=colname)

    return df_target


def writeffm(df,listkeys,filename):
    nrows = df.shape[0]
    sessions = {}

    ncore = multiprocessing.cpu_count()
    print("Using %d CPUs ..." % ncore)

    divider_ = divider(ncore,df.shape[0])

    def vecwrite(p,start,end,df,listkeys):
        block = ''
        for r in range(start,end):
            if r%100000 == 0 : print("Processing %d th row of %d th process ..." % (r,p))
            data = ''
            row = df.iloc[r].to_dict()
            data += str(int(row['label']))

            for f,fieldkeys in enumerate(listkeys):
                for colname,idx in fieldkeys.items():
                    featurestr = ' '+str(f)+':'+str(idx)+':'+str(int(row[colname]))
                    data += featurestr

            data += '\n'
            block += data

        return p, block

    vecresult = Parallel(n_jobs=ncore)(delayed(vecwrite)(p,divider_[p],divider_[p+1],df,listkeys) for p in range(ncore))
    vecresult.sort()

    text = ''
    for tup in vecresult:
        text += tup[1]

    with open(filename,'w+') as file:
        file.write(text)

    return
