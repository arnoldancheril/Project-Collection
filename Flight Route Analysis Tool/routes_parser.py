import pandas as pd

def parse_routes(file, out):
    file_in = open(file, 'r')
    routes_df = pd.read_csv(file_in, sep=",", header=None, 
                names = ["Airline","Airline ID","Source airport","Source airport ID", 
                        "Destination airport","Destination airport ID","Codeshare","Stops","Equipment"]
            )
    # print(routes_df)
    routes_df.drop(['Airline','Airline ID', 'Source airport',"Destination airport","Codeshare","Stops","Equipment"], axis=1,inplace=True)
    routes_df.to_csv(out ,index=False, header = False)

parse_routes('./routes.txt', './routes_parsed.txt')
