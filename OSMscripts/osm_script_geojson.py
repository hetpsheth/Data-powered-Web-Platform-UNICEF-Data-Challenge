import csv
import sys
import requests 
import json
import pandas as pd
import pyproj    
import shapely
import shapely.ops as ops
from shapely.geometry.polygon import Polygon
from functools import partial

filename = sys.argv[1]
outputFile = sys.argv[2]
final_data = []
with open(filename) as json_file:
    json_data = json.load(json_file)
    lines = json_data['features']
    header = ["codmun", "pharmacy", "hospital", "police", "school", "bank", "embassy", "tower","commercial", "industrial", "government", "supermarket", "park", "primary", "secondary", "unpaved", "paved"]
    final_data.append(header)
    URL = 'https://nominatim.openstreetmap.org/reverse'
    SEARCH_URL = 'https://nominatim.openstreetmap.org/search?'
    nodes={
            "amenity" : ["pharmacy", "hospital", "police", "school", "bank", "embassy"],
            "power" : ["tower"],
            "building" : ["commercial", "industrial", "government", "supermarket"],
            "leisure" : ["park"],
            }
    ways = {
            "highway" : ["primary", "secondary"],
            "surface" : ["unpaved", "paved"],
            } 
    overpass_url = "http://overpass-api.de/api/interpreter"
    for line in lines:
        codmun = int(line['properties']['CODMUN'])
        print(codmun)
        coordinates = line['geometry']['coordinates']
        polygon = ""
        points =[]
        for lvl1 in coordinates:
            for lvl2 in lvl1:
               
                if(len(lvl2)>300):
                    print("Very Long URI Coming up {}".format(len(lvl2)))
                    coord_count = 0
                    total_lat =0
                    total_long =0
                  
                    for lvl3 in lvl2:
                        if(coord_count == int(len(lvl2)/25)):
                            avg_lat = round(total_lat /coord_count, 4)
                            avg_long = round(total_long/coord_count, 4)
                            coordinate = "{} {}".format(avg_lat, avg_long)
                            polygon = polygon + " " + coordinate
                            points.append((round(float(lvl3[1]), 4), round(float(lvl3[0]), 4)))
                            coord_count = 0
                            total_lat =0
                            total_long =0
                        else:
                            total_lat = total_lat + round(float(lvl3[1]), 4)
                            total_long = total_long + round(float(lvl3[0]), 4)
                            coord_count = coord_count + 1
                    if(coord_count > 0):
                        avg_lat = total_lat /coord_count;
                        avg_long = total_long/coord_count;
                        coordinate = "{} {}".format(avg_lat, avg_long)
                        polygon = polygon + " " + coordinate
                        coordinate = 0
                    
                        
                else:
                    for lvl3 in lvl2:

                        coordinate = "{} {}".format(round(float(lvl3[1]), 4), round(float(lvl3[0]), 4))
                        polygon = polygon + " " + coordinate
                        points.append((round(float(lvl3[1]), 4), round(float(lvl3[0]), 4)))
               

        polygon = polygon[1:]
        '''
        geom = Polygon(points)
        geom_aea = ops.transform(
        partial(
            pyproj.transform,
            pyproj.Proj(init='EPSG:4326'),
            pyproj.Proj(
                proj='aea',
                lat1=geom.bounds[1],
                lat2=geom.bounds[3])),
        geom)

        area = round(geom_aea.area, 4)
        '''
        i=1
        
        data_row = []
        data_row.append(codmun)
        #data_row.append(area)
        for key, value in nodes.items():
            for val in value:
                query = """
                [out:json];
                (node['"""+ key +"""'='"""+val+"""'](poly:'"""+polygon+"""'); 
                );
                out count;
                """
                try:
                    response = requests.get(overpass_url, params={'data': query}, timeout=15)
                    cnt = response.json()['elements'][0]['tags']['nodes']
                    #print("{} {} = {}".format(key, val, cnt))
                    data_row.append(cnt)
                except json.decoder.JSONDecodeError as json_error:
                    print("json.decoder.JSONDecodeError Error")
                    data_row.append("null")
                except TimeoutError:
                    response = requests.get(overpass_url, params={'data': query}, timeout=25)
                    cnt = response.json()['elements'][0]['tags']['nodes']
                    data_row.append(cnt)



        for key, value in ways.items():
            for val in value:
                query = """
                [out:json];
                (way['"""+ key+"""'='"""+val+"""'](poly:'"""+polygon+"""'); 
                );
                out count;
                """
                try:
                    response = requests.get(overpass_url, params={'data': query})

                    cnt = response.json()['elements'][0]['tags']['ways']
                    #print("{} {} = {}".format(key, val, cnt))
                    data_row.append(cnt)
                except json.decoder.JSONDecodeError as json_error:
                    print("json.decoder.JSONDecodeError Error")
                    data_row.append("null")
        final_data.append(data_row)

with open(outputFile, 'w', newline='\n') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(final_data)

    





        
