from pathlib import Path
import pandas as pd 
import pickle 
from collections import defaultdict
import json 

DB_FOLDER = Path('data/db')
GOALS_FOLDER = Path('data/goals')

data = {
    'train',
    'test',
    'dev'
}

goals = []
for key in data:
    with open(GOALS_FOLDER / f'{key}_goals.json') as file:
        goals += list(map(lambda x:json.loads(x), file.readlines()))

with open(DB_FOLDER / 'restaurant_db_dict.pickle', 'rb') as file:
    slots = pickle.load(file)

informs = list(map(lambda x: x['inform_slots'], goals))

possible_restaurant_names = slots['restaurant_name']
possible_locations = slots['location']
possible_restaurants = [{'name': name, 'location': location} for name in possible_restaurant_names for location in possible_locations]
possible_restaurants_hash_table = {(i['name'], i['location']): defaultdict(list) for i in possible_restaurants}

informs_with_location_and_restaurant_name = list(filter(lambda x: all([x.get('location', None), x.get('restaurant_name', None)]), informs))
for i in informs_with_location_and_restaurant_name:
    key = (i['restaurant_name'], i['location'])

    if i.get('meal', None) and i['meal'] not in possible_restaurants_hash_table[key]['meal']:
        possible_restaurants_hash_table[key]['meal'].append(i['meal'])
    
    if i.get('category', None) and i['category'] not in possible_restaurants_hash_table[key]['category']:
        possible_restaurants_hash_table[key]['category'].append(i['category'])

    if i.get('rating', None) and i['rating'] not in possible_restaurants_hash_table[key]['rating']:
        possible_restaurants_hash_table[key]['rating'].append(i['rating'])

    if i.get('price_range', None) and i['price_range'] not in possible_restaurants_hash_table[key]['price_range']:
        possible_restaurants_hash_table[key]['price_range'].append(i['price_range'])

informs_with_only_location = list(filter(lambda x: all([x.get('location', None), not x.get('restaurant_name', None)]), informs))
for i in informs_with_only_location:
    for name in possible_restaurant_names:
        key = (name, i['location'])

        if i.get('meal', None) and i['meal'] not in possible_restaurants_hash_table[key]['meal'] and not possible_restaurants_hash_table[key]['meal'] and i['meal'] != 'dontcare':
            possible_restaurants_hash_table[key]['meal'].append(i['meal'])
        
        if i.get('category', None) and i['category'] not in possible_restaurants_hash_table[key]['category'] and not possible_restaurants_hash_table[key]['category'] and i['category'] != 'dontcare':
            possible_restaurants_hash_table[key]['category'].append(i['category'])

        if i.get('rating', None) and i['rating'] not in possible_restaurants_hash_table[key]['rating'] and not possible_restaurants_hash_table[key]['rating'] and i['rating'] != 'dontcare':
            possible_restaurants_hash_table[key]['rating'].append(i['rating'])

        if i.get('price_range', None) and i['price_range'] not in possible_restaurants_hash_table[key]['price_range'] and not possible_restaurants_hash_table[key]['price_range'] and i['price_range'] != 'dontcare':
            possible_restaurants_hash_table[key]['price_range'].append(i['price_range'])

informs_with_only_restaurant_name = list(filter(lambda x: all([not x.get('location', None), x.get('restaurant_name', None)]), informs))
for i in informs_with_only_restaurant_name:
    for name in possible_locations:
        key = (i['restaurant_name'], possible_locations)

        if i.get('meal', None) and i['meal'] not in possible_restaurants_hash_table[key]['meal'] and not possible_restaurants_hash_table[key]['meal'] and i['meal'] != 'dontcare':
            possible_restaurants_hash_table[key]['meal'].append(i['meal'])
        
        if i.get('category', None) and i['category'] not in possible_restaurants_hash_table[key]['category'] and not possible_restaurants_hash_table[key]['category'] and i['category'] != 'dontcare':
            possible_restaurants_hash_table[key]['category'].append(i['category'])

        if i.get('rating', None) and i['rating'] not in possible_restaurants_hash_table[key]['rating'] and not possible_restaurants_hash_table[key]['rating'] and i['rating'] != 'dontcare':
            possible_restaurants_hash_table[key]['rating'].append(i['rating'])

        if i.get('price_range', None) and i['price_range'] not in possible_restaurants_hash_table[key]['price_range'] and not possible_restaurants_hash_table[key]['price_range'] and i['price_range'] != 'dontcare':
            possible_restaurants_hash_table[key]['price_range'].append(i['price_range'])

possible_restaurants_hash_table = {key: dict(value) for key, value in possible_restaurants_hash_table.items()}
possible_restaurants_final = [{'restaurant_name': key[0], 'location': key[1], 'category': category, 'price_range': price_range, 'meal': meal, 'rating': rating} for key, value in possible_restaurants_hash_table.items() for category in value['category'] for price_range in value['price_range'] for meal in value['meal'] for rating in value['rating']]

possible_num_people = slots['num_people']
possible_date = slots['date']
possible_time = slots['time']

possible_reservations = [{'num_people': num_people, 'date': date, 'time': time, **possible_restaurant} for num_people in possible_num_people for date in possible_date for time in possible_time for possible_restaurant in possible_restaurants_final]
possible_reservations = {index: reservation for index, reservation in enumerate(possible_reservations)}

with open(DB_FOLDER / 'restaurant_db.pickle', 'wb') as file:
    pickle.dump(possible_reservations, file)