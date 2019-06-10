import numpy as np
import pandas as pd
from pathlib import Path
import click
from sklearn.utils import resample
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from . import dimensionality_reduction as dimred


current_directory = Path(__file__).absolute().parent
default_data_directory = current_directory.joinpath('..', '..', 'data')
@click.command()
@click.option('--data-path', default='./data/', help='Directory for the CSV files')

def main(data_path):
    # calculate path to files
    data_directory = Path(data_path) if data_path else default_data_directory
    train_csv = data_directory.joinpath('train.csv')
    test_csv = data_directory.joinpath('test.csv')
    meta_csv = data_directory.joinpath('item_metadata.csv')
    meta_dimred_csv = data_directory.joinpath('d5_ep300_datanorateclustering_firstattempt.csv')
    meta_encoded_csv = data_directory.joinpath('item_metadata_encoded.csv')

    print(f"Reading {meta_dimred_csv}...")
    df_dimred_items = pd.read_csv(meta_dimred_csv)
    print(f"Reading {meta_encoded_csv}...")
    df_items = pd.read_csv(meta_encoded_csv)



    print("dimred dimension",df_dimred_items.shape)
    print("original dimension",df_items.shape)
    print("MUST AGREE")
    print('Total:  ', df_items.shape[0])
    print('1 Star: ', df_items[df_items['1 Star']==1].shape[0]) 
    print('2 Star: ', df_items[df_items['2 Star']==1].shape[0]) 
    print('3 Star: ', df_items[df_items['3 Star']==1].shape[0]) 
    print('4 Star: ', df_items[df_items['4 Star']==1].shape[0]) 
    print('5 Star: ', df_items[df_items['5 Star']==1].shape[0]) 
    print('No Star Rating: ', df_items[(df_items['1 Star']==0) & (df_items['2 Star']==0) & (df_items['3 Star']==0) & (df_items['4 Star']==0) & (df_items['5 Star']==0)].shape[0])
    print('Other Star Rating ', df_items[((df_items['1 Star']==0) & (df_items['2 Star']==0) & (df_items['3 Star']==0) & (df_items['4 Star']==0) & (df_items['5 Star']==0)) & ((df_items['From 2 Stars']) | (df_items['From 3 Stars']) | (df_items['From 4 Stars']))].shape[0])

    df_exact_rated = df_items[(df_items['1 Star']==1) | (df_items['2 Star']==1) | (df_items['3 Star']==1) | (df_items['4 Star']==1) | (df_items['5 Star']==1)]
    df_exact_rated = df_exact_rated.reset_index(drop=True)
    print('Rated Subset (includes star rating): ',df_exact_rated.shape)
    df_exactr_dimr = df_dimred_items[(df_items['1 Star']==1) | (df_items['2 Star']==1) | (df_items['3 Star']==1) | (df_items['4 Star']==1) | (df_items['5 Star']==1)]
    df_exactr_dimr = df_exactr_dimr.reset_index(drop=True)
    print('Rated Subset Dimreduced (no stars): ',df_exactr_dimr.shape)

    print("training kmeans model...")
    kmeans = KMeans(n_clusters=5, random_state=0).fit(df_exactr_dimr)
    print("training completed!")

    df_rating_column = (df_exact_rated['1 Star']==1)*1 + (df_exact_rated['2 Star']==1)*2 + (df_exact_rated['3 Star']==1)*3 + (df_exact_rated['4 Star']==1)*4 + (df_exact_rated['5 Star']==1)*5
    score = np.zeros((5,5))
    
    estimated = kmeans.predict(df_exactr_dimr)
    real = df_rating_column

    for index in range(len(estimated)):
        score[estimated[index]-1, real[index]-1]+=1
    print(score)

    all_keys = ['reference', '1 Star', '2 Star', '3 Star', '4 Star', '5 Star', 
                'Accessible Hotel', 'Accessible Parking', 'Adults Only', 'Air Conditioning', 'Airport Hotel', 
                'Airport Shuttle', 'All Inclusive (Upon Inquiry)', 'Balcony', 'Bathtub', 'Beach', 'Beach Bar', 
                'Beauty Salon', 'Bed & Breakfast', 'Bike Rental', 'Boat Rental', 'Body Treatments', 
                'Boutique Hotel', 'Bowling', 'Bungalows', 'Business Centre', 'Business Hotel', 'Cable TV', 
                'Camping Site', 'Car Park', 'Casa Rural (ES)', 'Casino (Hotel)', 'Central Heating', 'Childcare', 
                'Club Hotel', 'Computer with Internet', 'Concierge', 'Conference Rooms', 'Convenience Store', 
                'Convention Hotel', 'Cosmetic Mirror', 'Cot', 'Country Hotel', 'Deck Chairs', 'Design Hotel', 
                'Desk', 'Direct beach access', 'Diving', 'Doctor On-Site', 'Eco-Friendly hotel', 
                'Electric Kettle', 'Excellent Rating', 'Express Check-In / Check-Out', 'Family Friendly', 'Fan', 
                'Farmstay', 'Fitness', 'Flatscreen TV', 'Free WiFi (Combined)', 'Free WiFi (Public Areas)', 
                'Free WiFi (Rooms)', 'Fridge', 'From 2 Stars', 'From 3 Stars', 'From 4 Stars', 'Gay-friendly', 
                'Golf Course', 'Good Rating', 'Guest House', 'Gym', 'Hairdresser', 'Hairdryer', 'Halal Food', 
                'Hammam', 'Health Retreat', 'Hiking Trail', 'Honeymoon', 'Horse Riding', 'Hostal (ES)', 'Hostel', 
                'Hot Stone Massage', 'Hotel', 'Hotel Bar', 'House / Apartment', 'Hydrotherapy', 'Hypoallergenic Bedding', 
                'Hypoallergenic Rooms', 'Ironing Board', 'Jacuzzi (Hotel)', "Kids' Club", 'Kosher Food', 'Large Groups', 
                'Laundry Service', 'Lift', 'Luxury Hotel', 'Massage', 'Microwave', 'Minigolf', 'Motel', 'Nightclub', 
                'Non-Smoking Rooms', 'On-Site Boutique Shopping', 'Openable Windows', 'Organised Activities', 
                'Pet Friendly', 'Playground', 'Pool Table', 'Porter', 'Pousada (BR)', 'Radio', 'Reception (24/7)', 
                'Resort', 'Restaurant', 'Romantic', 'Room Service', 'Room Service (24/7)', 'Safe (Hotel)', 'Safe (Rooms)', 
                'Sailing', 'Satellite TV', 'Satisfactory Rating', 'Sauna', 'Self Catering', 'Senior Travellers', 
                'Serviced Apartment', 'Shooting Sports', 'Shower', 'Singles', 'Sitting Area (Rooms)', 'Ski Resort', 'Skiing', 
                'Solarium', 'Spa (Wellness Facility)', 'Spa Hotel', 'Steam Room', 'Sun Umbrellas', 'Surfing', 
                'Swimming Pool (Bar)', 'Swimming Pool (Combined Filter)', 'Swimming Pool (Indoor)', 'Swimming Pool (Outdoor)', 
                'Szep Kartya', 'Table Tennis', 'Telephone', 'Teleprinter', 'Television', 'Tennis Court', 
                'Tennis Court (Indoor)', 'Terrace (Hotel)', 'Theme Hotel', 'Towels', 'Very Good Rating', 'Volleyball', 
                'Washing Machine', 'Water Slide', 'Wheelchair Accessible', 'WiFi (Public Areas)', 'WiFi (Rooms)']
    
    #keys connected to rating
    rating_keys = ['1 Star', '2 Star', '3 Star', '4 Star', '5 Star', 'From 2 Stars', 'From 3 Stars', 'From 4 Stars']
    no_rating_keys = [key for key in all_keys if key not in rating_keys]
    
    #define keys, that are a subjective rating like "4 Stars" or "Romantic Hotel"
    subjective_keys = ['1 Star', '2 Star', '3 Star', '4 Star', '5 Star', 'From 2 Stars', 'From 3 Stars', 'From 4 Stars', 
                       'Excellent Rating', 'Gay-friendly', 'Honeymoon', 'Satisfactory Rating', 'Senior Travellers', 
                       'Very Good Rating']

    objective_keys = [key for key in all_keys if key not in subjective_keys]

    #NOTE: Only train with data before splitting point as data set too huge
    #TODO: send different df_items: (No stars, no subject, etc...)
    #TODO: different encoding dimensions
    #TODO: different number of epochs
    #TODO: maybe try larger training sample than 4000, but looking at online samples they 
    #      don't use more than around 4000 for training of encoder
    #FUNCTION: dimred.reduce(dataframe, splitting point, encod_dim, nb_epoch)

    #Option 1: small test to see if everything is working (not full itemset)
    # encoded_item = dimred.reduce(df_items.loc[:,no_rating_keys][0:10000], 1000, 20, 1)
    # print(f"Writing to {dimred_encoded_item_csv} ...")
    # encoded_item.to_csv(dimred_encoded_item_csv, index=False)

    # #Option 2: one full run with one set of reasonable parameters
    # encoded_item = dimred.reduce(df_items, 4000, 20, 200)
    # print(f"Writing to {dimred_encoded_item_csv} ...")
    # encoded_item.to_csv(dimred_encoded_item_csv, index=False)

    # #Option 3: LONG: create a set of useful datasets for analysis with kmeans
    # counter = 0
    # encod_dim = [5,10]
    # nmb_epochs = [300]
    # datasets = [[df_items.loc[:,no_rating_keys],"norate"]]
    # for dim in encod_dim:
    #     for epoc in nmb_epochs:
    #         for data in datasets:
    #             counter += 1
    #             print(f"working on iteration {counter} of {len(encod_dim)*len(nmb_epochs)*len(datasets)}")
    #             csv_name = "d" + str(dim) + "_ep" + str(epoc) + "_data" + data[1] + "clustering_firstattempt.csv"
    #             print(csv_name)
    #             dimred_encoded_item_csv = data_directory.joinpath(csv_name)
    #             encoded_item = dimred.reduce(data[0], 4000, dim, epoc)
    #             encoded_item.to_csv(dimred_encoded_item_csv, index=False)


if __name__ == '__main__':
    main()