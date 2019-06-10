import numpy as np
import pandas as pd
from pathlib import Path
import click
from sklearn.utils import resample
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.cm as cm
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
    subm_csv = data_directory.joinpath('submission_popular.csv')

    meta_encoded_csv = data_directory.joinpath('item_metadata_encoded.csv')
    dimred_encoded_item_csv = data_directory.joinpath('dimred_encoded_item.csv')

    all_keys = ['1 Star', '2 Star', '3 Star', '4 Star', '5 Star',
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
    exact_ratings = ['1 Star', '2 Star', '3 Star', '4 Star', '5 Star']
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

    print(f"Reading {meta_encoded_csv}...")
    df_items = pd.read_csv(meta_encoded_csv)
    #TODO: reinsert line for debugging
    #df_items = df_items.head(20000)

    print("Extracting hotels with ratings ...")
    df_withrate = df_items[(df_items['1 Star']==1) | (df_items['2 Star']==1) | (df_items['3 Star']==1) | (df_items['4 Star']==1) | (df_items['5 Star']==1)]
    df_withrate = df_withrate.reset_index(drop=True)
    
    #FUNCTION: dimred.reduce(dataframe, splitting point, encod_dim, nb_epoch)
    #TODO: MAKE SURE TO HAVE FOLDERS 1 TO X IN DATA
    run_configuration = [[1,objective_keys, 10000, 20, 300],
                         [2,objective_keys, 10000, 10, 300],
                         [3,no_rating_keys, 10000, 20, 300],
                         [4,no_rating_keys, 10000, 10, 300]]

    for run in run_configuration:
        dimred_encoded_item_csv = data_directory.joinpath(str(run[0])).joinpath('dimred_encoded_item'+str(run[0])+'.csv')
        try:
            dimred_encoded_item_csv.resolve(strict=True)
        except FileNotFoundError:
            #TODO: NAMING OF FILES AND PATHS
            #only reduce dim of data with excact ratings, as we can only analyze these data points later
            encoded_item = dimred.reduce(df_withrate.loc[:,run[1]], run[2], run[3], run[4])
            print(f"Writing to {dimred_encoded_item_csv} ...")
            encoded_item.to_csv(dimred_encoded_item_csv, index=False)
        else:
            print("Dimred encoded metadata file found ...")
            print(f"Reading {dimred_encoded_item_csv} ...")
            encoded_item = pd.read_csv(dimred_encoded_item_csv)

        print("Training K-means clustering ...")

        kmeans = KMeans(n_clusters=5,random_state=0).fit(encoded_item)
        prediction = kmeans.predict(encoded_item)
        rates = df_withrate[rating_keys].apply(dimred.undoonehot,axis=1)

        print("Processing T-SNE ...")

        tsne = TSNE(n_components=2, perplexity=40, n_iter=300)
        tsne_results = tsne.fit_transform(encoded_item)

        plt.figure()
        colors = cm.rainbow(np.linspace(0, 1, 5))
        ax1 = plt.subplot(1,2,1)
        ax1.scatter(tsne_results[:,0],tsne_results[:,1],c=colors[prediction],s=1)
        ax2 = plt.subplot(1,2,2)
        ax2.scatter(tsne_results[:,0],tsne_results[:,1],c=colors[rates],s=1)
        ax2.legend()
        plt.draw()
        tsne_fig = data_directory.joinpath(str(run[0])).joinpath('tsne'+str(run[0])+'.png')
        plt.savefig(tsne_fig, dpi=300, bbox_inches='tight')



if __name__ == '__main__':
    main()
