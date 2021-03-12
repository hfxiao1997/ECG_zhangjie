import glob
import os
import pandas as pd

input_path = 'data'
csv_list = []
for csv_file in os.listdir(input_path):
    csv_path = os.path.join(input_path, csv_file)
    csv_list.append(csv_path)

outputfile = 'all_in_one/result.csv'

print(csv_list)
filepath = csv_list[0]
df = pd.read_csv(filepath)
df = df.to_csv(outputfile, index=False)

for i in range(1, len(csv_list)):
    filepath = csv_list[i]
    df = pd.read_csv(filepath)
    df = df.to_csv(outputfile, index=False, header=False, mode='a+')