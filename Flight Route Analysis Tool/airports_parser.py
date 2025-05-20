import pandas as pd

def parse_airports(file, out):
    file_in = open(file, 'r', encoding="utf8")
    airports_df = pd.read_csv(file_in, sep=",", header=None, 
                names = ['Airport ID', 'Name', 'City', 'Country', 'IATA', 'ICAO', 'Latitude', 'Longitude', 'Altitude', 'Timezone', 'DST', 'Tz', 'Type', 'Source'])
    airports_df.drop(['Name', 'City', 'Country', 'IATA', 'ICAO', 'Altitude', 'Timezone', 'DST', 'Tz', 'Type', 'Source'], axis=1,inplace=True)
    airports_df.to_csv(out ,index=False, header = False)

parse_airports('./airports.txt', './airports_parsed2.txt')