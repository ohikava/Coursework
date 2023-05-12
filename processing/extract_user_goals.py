from pathlib import Path
import pandas as pd 
import json 

DATA_OUPUT_FOLDER = Path('data/goals')
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

def generate_simple_user_goals(x):
    state = x[1]['dialogue_state'] if len(x) > 1 else x[0]['dialogue_state']

    res = {}
    res['inform_slots'] = {}
    res['request_slots'] = {}
    for i in state:
        slot = i['slot']
        value = i['value']
        res['inform_slots'][slot] = value 
    return res

def get_user_goals(df):
    turns = df['turns']
    return turns.apply(generate_simple_user_goals).to_list()

def clear_dontcare_slots(goals):
    return list(map(lambda goal: {**goal, 'inform_slots': {k:v for k, v in goal['inform_slots'].items() if v != 'dontcare'}}, goals))

def clear_empty_goals(goals):
    return list(filter(lambda x: x['inform_slots'], goals))

def data_processing_pipeline(df):
    goals = get_user_goals(df)
    goals = clear_dontcare_slots(goals)
    goals = clear_empty_goals(goals)
    return goals

goals = {k:data_processing_pipeline(v) for k, v in data.items()}

for key in goals:
    print(len(goals[key]), end=', ')

for k, v in goals.items():
    with open(DATA_OUPUT_FOLDER / f'{k}_goals.json', 'w', encoding='utf-8') as file:
        for goal in v:
            file.write(json.dumps(goal) + '\n')