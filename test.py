#!/usr/bin/env python3 
import json 


with open('dataset_folder/response.json', 'r') as f:
    data = json.load(f)

print(data['good_mood'][1])



