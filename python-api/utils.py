"""
Utils.py contains classes/functions that filter/extract data from the excel file 
Created by: Shantanu Mantri
"""
import pandas as pd
import numpy as np 
import csv

class DataReader:
	
	def __init__(self):
		"""
		Constructor for Data Reader, which is used as an abstraction class
		so that no pandas functions need to be used in the front-end
		"""
		self.country_frame = pd.read_csv("../data/country_new.csv", index_col = None)
		self.mobility_frame = pd.read_csv("../data/mobility.csv", index_col = None)
		self.final_data ={}
	
	def get_flow(self, country, o_codmun, d_codmun):
		"""
		Gets the flow from one country to another
		Parameters
		----------
		country:   String   the country
		o_codmun : Integer  the origin district's code
		d_conmun : Integer  the destination district's code
		
		Returns
		-------
		List : A list of size 2 with the flows for 2016 and 2017
		"""
		frame = self.mobility_frame
		df_filtered = frame[(frame['Country'] == country) & (frame['origin_codmun'] == o_codmun) & (frame['destination_codmun'] == d_codmun)]
		return [df_filtered.iloc[0]['flow_2016'], df_filtered.iloc[0]['flow_2017']]

	def get_inflow(self, country, d_codemun):
		"""
		Gets the inward flow for a particular district
		Parameters
		----------
		country:   String   the country
		d_conmun : Integer  the district's code
		
		Returns
		-------
		DataFrame : A frame with all inflows for 2016 and 2017
		"""
		frame = self.mobility_frame
		df_filtered = frame[(frame['Country'] == country) & (frame['destination_codmun'] == d_codemun)]
		return df_filtered[['flow_2016', 'flow_2017']]

	def get_outflow(self, country, o_codemun):
		"""
		Gets the outward flow for a particular district
		Parameters
		----------
		country:   String   the country
		o_conmun:  Integer  the district's code
		
		Returns
		-------
		DataFrame : A frame with all outflows for 2016 and 2017
		"""
		frame = self.mobility_frame
		df_filtered = frame[(frame['Country'] == country) & (frame['origin_codmun'] == o_codemun)]
		return df_filtered[['flow_2016', 'flow_2017']]
		
	def get_average_inflow(self, country, d_codemun):
		"""
		Gets the average inward flow for a particular district
		Parameters
		----------
		country:   String   the country
		d_conmun : Integer  the district's code
		
		Returns
		-------
		List : A list of size 2 for average inflows for 2016 and 2017
		"""
		frame = self.mobility_frame
		df_filtered = frame[(frame['Country'] == country) & (frame['destination_codmun'] == d_codemun)]
		df_filtered = df_filtered[['flow_2016', 'flow_2017']]
		df_mean = df_filtered.mean()
		return [df_mean.loc['flow_2016'], df_mean.loc['flow_2017']]


	def get_average_outflow(self, country, o_codemun):
		"""
		Gets the average outward flow for a particular district
		Parameters
		----------
		country:   String   the country
		o_conmun : Integer  the district's code
		
		Returns
		-------
		List : A list of size 2 for average outflows for 2016 and 2017
		"""
		frame = self.mobility_frame
		df_filtered = frame[(frame['Country'] == country) & (frame['origin_codmun'] == o_codemun)]
		df_filtered = df_filtered[['flow_2016', 'flow_2017']]
		df_mean = df_filtered.mean()
		return [df_mean.loc['flow_2016'], df_mean.loc['flow_2017']]

	def get_hdi(self, codmun, estimated=False, year=2016):
		"""
		Gets the HDI given for a given district
		Parameters
		----------
		codmun:    Integer   the district code
		estimated: Boolean   whether or not it's the actual HDI or an estimate
		year:      Integer   The year to get the HDI for 
		
		Returns
		-------
		Pandas DataFrame  
		"""
		frame = self.country_frame
		if estimated:
			return frame.loc[codmun]['hdi_estimated_' + str(year)]
		return frame[['hdi']].loc[codmun]

	def get_cols_country(self, cols, codmun):
		"""
		Gets columns from the country csv file for a given district
		Parameters
		----------
		cols:   List     the list of columns wanted
		codmun: Integer  the district's code
		
		Returns
		-------
		Pandas DataFrame    
		"""
		frame = self.country_frame
		return frame[(frame["codmun"] == codmun)][cols]

	def get_cols_mobility(self, cols, o_codmun, d_codmun):
		"""
		Gets columns from the mobility csv file for a given district
		Parameters
		----------
		cols:     List     the list of columns wanted
		o_codmun: Integer  the district's code origin
		d_codmun: Integer  the district's code destination
		
		Returns
		-------
		Pandas DataFrame    
		"""
		frame = self.mobility_frame
		return frame[(frame["origin_codmun"] == o_codmun) & (frame["destination_codmun"] == d_codmun)][cols]

	def create_input_data(self):
		"""
		Creates input data necessary in a format appropriate for the model.
		Returns
		-------
		Pandas DataFrame: A dataframe of the input data
		"""
		frame = self.country_frame
		mob_frame = self.mobility_frame

		in_frame = mob_frame.groupby(['Country', 'origin_codmun'])[['flow_2016', 'flow_2017']].mean()
		out_frame = mob_frame.groupby(['Country', 'destination_codmun'])[['flow_2016', 'flow_2017']].mean()

		
		frame['average_inflow_2016'] = None  
		frame['average_outflow_2016'] = None
		frame['average_inflow_2017'] = None  
		frame['average_outflow_2017'] = None

		inflow_16 = []
		inflow_17 = []
		outflow_16 = []
		outflow_17 = []

		valid_incodes = in_frame.index.levels
		valid_outcodes = out_frame.index.levels

		for idx in range(len(frame)):

			row = frame.iloc[idx]
			country = row.loc['Country']
			codmun = row.loc['codmun']

			try:
				in_row = in_frame.loc[country, codmun]
				inflow_16.append(in_row.loc['flow_2016'])
				inflow_17.append(in_row.loc['flow_2017'])
			except:
				inflow_16.append(None)
				inflow_17.append(None)

			try:
				out_row = out_frame.loc[country, codmun]
				outflow_16.append(out_row.loc['flow_2016'])
				outflow_17.append(out_row.loc['flow_2017'])
			except:
				outflow_16.append(None)
				outflow_17.append(None)
			
		frame['average_inflow_2016'] = inflow_16 
		frame['average_outflow_2016'] = outflow_16
		frame['average_inflow_2017'] = inflow_17
		frame['average_outflow_2017'] = outflow_17
		frame.fillna(0, inplace=True)
		
		return frame

	def parse_country(self):
		with open('../data/country_hdi.csv', encoding="UTF-8") as csv_file:
			csv_reader = csv.reader(csv_file, delimiter=',')
			line_count = 0
			zoom_level = {
				'brazil': 1,
				'costarica': 4,
				'nepal': 4,
				'pakistan': 3,
				'poland': 4,
				'mexico': 2,
				'colombia': 3, 
				'nigeria': 3,
			}
			for row in csv_reader:
				if line_count==0:
					pass
				else:
					if row[3]:
						country = row[3]
						country_slug = ''.join(country.split()).lower()
						self.final_data[country_slug] = {}
						self.final_data[country_slug]["zoom"] = zoom_level.get(country_slug)
						self.final_data[country_slug]["latitude"] = row[1]
						self.final_data[country_slug]["longitude"] = row[2]
						self.final_data[country_slug]["hdi"] = str(0)
						self.final_data[country_slug]["country_name"] = country
						self.final_data[country_slug]["cities"] = {}
						lower_country = row[3].lower()
						with open('../data/country_hdi.csv', encoding="UTF-8") as second_fp:
							read_csv = csv.reader(second_fp, delimiter=',')
							for each_row in read_csv:
								if lower_country ==each_row[4].lower():
									self.final_data[country_slug]["hdi"] = each_row[5]
				line_count += 1
		count = 0
		for key, val in self.final_data.items():
			if 'HDI' in val.keys():
				count+=1
	
	def parse_city(self):
		with open('../data/city.csv', encoding="UTF-8") as csv_file:
			csv_reader = csv.reader(csv_file, delimiter=',')
			line_count = 0
			for row in csv_reader:
				if line_count==0:
					pass
				else:
					if row[4] in self.final_data.keys():
						country = row[4]
						codmun =row[0]
						city_name =row[3]
						self.final_data[country]['cities'][codmun] = {}
						self.final_data[country]['cities'][codmun]['latitude'] = row[1]
						self.final_data[country]['cities'][codmun]['longitude'] = row[2]
						self.final_data[country]['cities'][codmun]['city_name'] = city_name
				line_count +=1

	def parse_features(self):
		with open('../data/country_new.csv', encoding="UTF-8") as csv_file:
			csv_reader = csv.reader(csv_file, delimiter=',')
			line_count = 0
			for row in csv_reader:
				if line_count==0:
					pass
				else:
					country =row[0]
					codmun = row[1]
					if country in self.final_data.keys():
						if self.final_data[country]['cities'] and codmun in self.final_data[country]['cities'].keys():
							self.final_data[country]['cities'][codmun]['real_hdi'] = row[2]
							self.final_data[country]['cities'][codmun]['popularity_2016'] = row[7]
							self.final_data[country]['cities'][codmun]['popularity_2017'] = row[8]
							self.final_data[country]['cities'][codmun]['activity_2016_h0-5'] = row[9]
							self.final_data[country]['cities'][codmun]['activity_2016_h6-11'] = row[10]
							self.final_data[country]['cities'][codmun]['activity_2016_h12-17'] = row[11]
							self.final_data[country]['cities'][codmun]['activity_2016_h18-23'] = row[12]
							self.final_data[country]['cities'][codmun]['activity_2017_h0-5'] = row[13]
							self.final_data[country]['cities'][codmun]['activity_2017_h6-11'] = row[14]
							self.final_data[country]['cities'][codmun]['activity_2017_h12-17'] = row[15]
							self.final_data[country]['cities'][codmun]['activity_2017_h18-23'] = row[16]
							self.final_data[country]['cities'][codmun]['bank'] = row[17]
							self.final_data[country]['cities'][codmun]['government'] = row[20]
							self.final_data[country]['cities'][codmun]['pharmacy'] = row[25]
							self.final_data[country]['cities'][codmun]['secondary'] = row[29]
				line_count+=1
	
	def parse_prediction(self):
		with open('../data/predictions_2016.csv', encoding="UTF-8") as csv_file:
			csv_reader = csv.reader(csv_file, delimiter=',')
			line_count = 0
			for row in csv_reader:
				if line_count==0:
					pass
				else:
					country =row[1]
					codmun = row[0]
					if country in self.final_data.keys():
						if self.final_data[country]['cities'] and codmun in self.final_data[country]['cities'].keys():
							self.final_data[country]['cities'][codmun]['predicted_hdi_2016'] = row[2]
				line_count+=1
		with open('../data/predictions_2017.csv', encoding="UTF-8") as csv_file:
			csv_reader = csv.reader(csv_file, delimiter=',')
			line_count = 0
			for row in csv_reader:
				if line_count==0:
					pass
				else:
					country =row[1]
					codmun = row[0]
					if country in self.final_data.keys():
						if self.final_data[country]['cities'] and codmun in self.final_data[country]['cities'].keys():
							self.final_data[country]['cities'][codmun]['predicted_hdi_2017'] = row[3]
				line_count+=1



#Example
#d = DataReader()
#print(d.create_input_data())
#print(d.get_cols_mobility(["flow_2016"], 1939, 1939))
#print(d.get_cols_country(["hdi"], 1939))
#print(d.get_hdi(1939))
#from matplotlib import pyplot as plt
#plt.style.use('dark_background')
#frame = pd.read_csv("../data/country.csv", index_col = None)
#frame = frame[(frame["codmun"] == 91) & (frame["Country"] == "pakistan")][["activity_2016_h0-5","activity_2016_h6-11","activity_2016_h12-17","activity_2016_h18-23"]]
#x_labels = frame.columns.tolist()
#fig, ax = plt.subplots(nrows=1, ncols=1)
#ax.set_ylabel('Tweet Frequency')
#ax.set_xlabel("Time of day (0 - 23 hours)")
#ax.set_title("Tweet Activity Frequency in Pakistan Municipality 91")
#red, = ax.plot(np.arange(4), frame.iloc[0].tolist(), 'r')
#green = ax.bar(np.arange(4), frame.iloc[0].tolist(), 0.25)
#ax.set_xticks(np.arange(4))
#ax.set_xticklabels(["h0-5","h6-11","h12-17","h18-23"], rotation='vertical')
#plt.tight_layout()
#fig.savefig("pakistan_twitter")

"""
###merge geodata###
cr = pd.read_csv("../data/costarica_geo.csv", index_col = None)
cr['Country'] = "costarica"
#mexico = pd.read_csv("../data/mexico_geo.csv", index_col = None)
#mexico['Country'] = "mexico"
nepal = pd.read_csv("../data/nepal_geo.csv", index_col = None)
nepal['Country'] = "nepal"
nigeria = pd.read_csv("../data/nigeria_geo.csv", index_col = None)
nigeria['Country'] = "nigeria"
pakistan = pd.read_csv("../data/pakistan_geo.csv", index_col = None)
pakistan['Country'] = "pakistan"
poland = pd.read_csv("../data/poland_geo.csv", index_col = None)
poland['Country'] = "poland"
colombia = pd.read_csv("../data/columbia_geo.csv", index_col = None)
colombia['Country'] = "colombia"
brazil = pd.read_csv("../data/brazil_final.csv", index_col = None)
brazil['Country'] = "brazil"
frames = [cr, nepal, nigeria, pakistan, poland, colombia, brazil]
geo_frame = pd.concat(frames)
country_frame = pd.read_csv("../data/country.csv", index_col = None)
print("CF Before Merge: ", len(country_frame))
final_frame = pd.merge(country_frame, geo_frame, how='left', left_on=['Country', 'codmun'], right_on=['Country', 'codmun'])
#final_frame.fillna(-1, inplace=True)
#final_frame.dropna(inplace=True) 
kek = final_frame[["bank", "commercial", "embassy", "government", "hospital", "industrial",
"park", "paved", "pharmacy", "police", "primary", "school", "secondary", "supermarket",
"tower", "unpaved"]].isna().sum(axis=1)
count = 0
hdi_l = [] #<= 0.5
hdi_m = [] # > 0.5 && <=0.7
hdi_h = [] # > 0.7
l_count = 0
m_count = 0
h_count = 0
for i in range(len(kek)):
	hdi = final_frame.iloc[i]['hdi']
	if hdi <= 0.5:
		l_count += 1
	elif hdi > 0.5 and hdi <= 0.7:
		m_count += 1
	else: 
		h_count += 1	
	if kek[i] == 16:
		hdi = final_frame.iloc[i]['hdi']
		if hdi <= 0.5:
			hdi_l.append(hdi)
		elif hdi > 0.5 and hdi <= 0.7:
			hdi_m.append(hdi)
		else: 
			hdi_h.append(hdi)
		
print("Low: ", len(hdi_l)/l_count, np.sum(hdi_l)/len(hdi_l))
print("Med: ", len(hdi_m)/m_count, np.sum(hdi_m)/len(hdi_m))
print("High: ", len(hdi_h)/h_count, np.sum(hdi_h)/len(hdi_h))

final_frame.drop_duplicates(inplace=True)
print("CF After Merge: ", len(final_frame))
#final_frame.to_csv("../data/country_new.csv", index=False)
#now we join by country and codmun
"""