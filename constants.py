# Special slot values (for reference)
PLACEHOLDER = 'PLACEHOLDER'  # For informs
UNK = 'UNK'  # For requests
ANYTHING = 'ANYTHING'  # means any value works for the slot with this value
NO_MATCH = 'no match available'  # When the intent of the agent is match_found yet no db match fits current constraints

# Possible diaacts
INFORM = 'inform'
REQUEST = 'request'
THANKS = 'thanks'
REJECT = 'reject'
DONE = 'done'
MATCH_FOUND = 'match_found'

# These are used for both constraint check AND success check in usersim
FAIL = -1
NO_OUTCOME = 0
SUCCESS = 1

# Possible moods of bot
TRAIN = 'TRAIN'
TEST = 'TEST'
DEV = 'DEV'

#######################################
# Usersim Config
#######################################
# Used in EMC for intent error (and in user)
usersim_intents = [INFORM, REQUEST, THANKS, REJECT, DONE]

# The goal of the agent is to inform a match for this key
usersim_default_key = 'reservation_id'

# Required to be in the first action in inform slots of the usersim if they exist in the goal inform slots
usersim_required_init_inform_keys = ['restaurant_name']

#######################################
# Agent Config
#######################################

# Possible inform and request slots for the agent
agent_inform_slots = ["restaurant_name", "date", "time", "meal", "location", "price_range", "category", "rating", usersim_default_key]

agent_request_slots = ["restaurant_name", "date", "time", "meal", "location", "price_range", "category", "rating", "num_people", usersim_default_key]

# Possible actions for agent
agent_actions = [
    {'intent': DONE, 'inform_slots': {}, 'request_slots': {}},  # Triggers closing of conversation
    {'intent': MATCH_FOUND, 'inform_slots': {}, 'request_slots': {}}
]
for slot in agent_inform_slots:
    # Must use intent match found to inform this, but still have to keep in agent inform slots
    if slot == usersim_default_key:
        continue
    agent_actions.append({'intent': INFORM, 'inform_slots': {slot: PLACEHOLDER}, 'request_slots': {}})
for slot in agent_request_slots:
    agent_actions.append({'intent': REQUEST, 'inform_slots': {}, 'request_slots': {slot: UNK}})

# Rule-based policy request list
rule_requests = ['restaurant_name', 'time', 'location', 'date', 'num_people']

# These are possible inform slot keys that cannot be used to query
no_query_keys = ['num_people', usersim_default_key]

#######################################
# Global config
#######################################

# All possible intents (for one-hot conversion in ST.get_state())
all_intents = [INFORM, REQUEST, DONE, MATCH_FOUND, THANKS, REJECT]

# All possible slots (for one-hot conversion in ST.get_state())
all_slots = ["restaurant_name", "date", "time", "meal", "location", "price_range", "category", "rating", "num_people", usersim_default_key]
