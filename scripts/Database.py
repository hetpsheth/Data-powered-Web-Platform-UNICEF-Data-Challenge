import sqlite3
import csv

class Database:

	def __init__(self, name):
		self.conn = sqlite3.connect(name)

	def create_table_with_csv(self, csvfile, rows, table_name):
		cur = self.conn.cursor()
		body_string = "(" + ",".join([row[0] + " " + row[1] for row in rows]) + ")"

		sql_string = "CREATE TABLE IF NOT EXISTS " + table_name + " " + body_string  	

		cur.execute(sql_string)

		f = open(csvfile, 'r')
		next(f, None)
		reader = csv.reader(f)

		val_string = "(" + ",".join(["?" for i in range(len(rows))]) + ")"

		sql_string = "INSERT INTO " + table_name + " VALUES " + val_string

		for r in reader:
			cur.execute(sql_string, r)

		f.close()
		self.conn.commit()
		cur.close()

	def select_from_table(self, rows, table_name):
		cur = self.conn.cursor()
		sql_string = "SELECT " + ",".join(rows) + " FROM " + table_name

	def close(self):
		self.conn.close()

input_rows = [("country", "text"),
			 ("codmun", "text"),
			 ("hdi", "float"),
			 ("hdi_estimated_2016", "float"),
			 ("hdi_estimated_2017", "float"),
			 ("entropy_2016", "float"),
			 ("entropy_2017", "float"),
			 ("popularity_2016", "float"),
			 ("popularity_2017", "float"),
			 ("activity_2016_h0_5", "float"),
			 ("activity_2016_h6_11", "float"),
			 ("activity_2016_h12_17", "float"),
			 ("activity_2016_h18_23", "float"),
			 ("activity_2017_h0_5", "float"),
			 ("activity_2017_h6_11", "float"),
			 ("activity_2017_h12_17", "float"),
			 ("activity_2017_h18_23", "float"),
			 ("bank", "float"),
			 ("commercial", "float"),
			 ("embassy", "float"),
			 ("government", "float"),
			 ("hospital", "float"),
			 ("industrial", "float"),
			 ("park", "float"),
			 ("paved", "float"),
			 ("pharmacy", "float"),
			 ("police", "float"),
			 ("primarie", "float"),
			 ("school", "float"),
			 ("secondary", "float"),
			 ("supermarket", "float"),
			 ("tower", "float"),
			 ("unpaved", "float")]

d = Database("poverty.db")
try:
	d.create_table_with_csv("../data/country_new.csv", input_rows, "countries")
except:
	pass
	
rows = [("origin_codmun", "integer"),
		("destination_codmun", "integer"),
		("flow_2016", "float"),
		("flow_2017", "float"), 
		("Country", "float")]

d.create_table_with_csv("../data/mobility.csv", , "mobility")