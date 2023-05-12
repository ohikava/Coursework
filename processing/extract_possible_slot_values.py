from pathlib import Path
import pandas as pd 
import pickle 
from collections import defaultdict
from tqdm import tqdm 

DATA_OUPUT_FOLDER = Path('data/db')
DATA_INPUT_FOLDER = Path('data/raw')

# Raw data
train = pd.read_json(DATA_INPUT_FOLDER / 'train.json')
test = pd.read_json(DATA_INPUT_FOLDER / 'test.json')
dev = pd.read_json(DATA_INPUT_FOLDER / 'dev.json')

data = {
    'train': train,
    'test': test,
    'dev': dev
}

def get_dialogue_state(df):
    turns = df['turns'].explode()
    dialogue_state = turns.map(lambda x: x['dialogue_state'])
    return dialogue_state.dropna()

def get_exploded_dialogue_slots(dialogue_state):
    dialogue_dicts = dialogue_state.explode()
    exploded_dialogue_slots = dialogue_dicts.apply(pd.Series).drop(0, axis=1).dropna()
    return exploded_dialogue_slots

slots = defaultdict(set)

for _, df in tqdm(data.items()):
    dialogue_state = get_dialogue_state(df)
    exploded_dialogue_slots = get_exploded_dialogue_slots(dialogue_state)
    possible_slots = exploded_dialogue_slots['slot'].unique()

    for key in possible_slots:
        slots[key] = slots[key].union(exploded_dialogue_slots[exploded_dialogue_slots['slot'] == key]['value'].unique())
        
for key in slots:
    slots[key] = [i for i in slots[key] if i != 'dontcare']

with open(DATA_OUPUT_FOLDER / 'restaurant_db_dict.pickle', 'wb') as file:
    pickle.dump(dict(slots), file)