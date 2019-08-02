import csv
import sys
import requests 
import json
import pandas as pd

filename = sys.argv[1]
outputFile = sys.argv[2]
country = sys.argv[3]
final_data = []
threshold = sys.argv[4]
with open(outputFile, mode='w') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        next(csv_reader)
        #URL = 'https://nominatim.openstreetmap.org/reverse'
        URL ='http://open.mapquestapi.com/nominatim/v1/reverse.php'
        SEARCH_URL = 'https://nominatim.openstreetmap.org/search?'
        i=1
        header = ["codmun", "latitude", "longitude", "county", "country"]
        writer.writerow(header)
        line = 0 
        for row in csv_reader:
            data_row = []
            line = line + 1
            if (line > int(threshold)):
                tranlate_api = "https://translation.googleapis.com/language/translate/v2"
                longitude = row[0]
                latitude = row[1]
                codmun = row[2]
            
                PARAMS = {'key': 'scgrVPfaUoji6AloYXb3phovO87G7sWJ', 'format': 'jsonv2', 'lat':latitude, 'lon': longitude}

                response = requests.get(url = URL, params = PARAMS) 
                print(response)
                data = response.json()

                
                #state = data['address']['state']
                #country = data['address']['country']
                
                if 'county' in data['address']:
                    county = data['address']['county']
                elif 'region' in data['address']:
                    county = data['address']['region']
                else:
                    county="null"
                '''
                if(county not in english):
                    LANG_PARAMS = {'q': county, 'target':'en', 'format': 'text'}
                    county_response = requests.post(url = tranlate_api, params = PARAMS)
                print(county)
                '''
                data_row.append(codmun)
                data_row.append(latitude)
                data_row.append(longitude)
                data_row.append(county)
                data_row.append(country)
                writer.writerow(data_row)







            
