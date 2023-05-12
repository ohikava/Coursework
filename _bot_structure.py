import json
import os
import pickle
from dqn_agent import DQNAgent
from error_model_controller import ErrorModelController
from constants import TRAIN, TEST, DEV
from state_tracker import StateTracker
from usersim import UserSimulator
from utils import remove_empty_slots


class BotStructure:
    def __init__(self, constants, Agent=None):
        self.constants = constants 
        
        file_path_dict = constants['db_file_paths']
        # User goals for train, dev, test correspondingly
        self.train_file_path = file_path_dict['user_goals']
        self.dev_file_path = file_path_dict['dev_goals']
        self.test_file_path = file_path_dict['test_goals']
        
        self.database = pickle.load(open(file_path_dict['database'], 'rb'))
        self.db_dict = pickle.load(open(file_path_dict['dict'], 'rb'))

        # Load run constants
        run_dict = constants['run']
        self.use_usersim = run_dict['usersim']
        self.warmup_mem = run_dict['warmup_mem']
        self.num_ep_train = run_dict['num_ep_run']
        self.num_ep_test = run_dict['num_ep_test']
        self.train_freq = run_dict['train_freq']
        self.max_round_num = run_dict['max_round_num']
        self.success_rate_threeshold = run_dict['success_rate_threshold']
        self.log_train_path = run_dict['train_log_path']
        self.log_dev_path = run_dict['dev_log_path']
        self.log_test_path = run_dict['test_log_path']    
        
        self.early_stopping_delay = run_dict['early_stopping_delay']  
        self.early_stopping_rounds= run_dict['early_stopping_rounds']
        
        if self.early_stopping_rounds:
            self.rounds_without_eval_improvements = 0
        # Turn on train mode by default
        self.train()

        # Clean DB
        remove_empty_slots(self.database)

        self.emc = ErrorModelController(self.db_dict, constants)
        self.state_tracker = StateTracker(self.database, constants)
        
        if Agent:
            self.dqn_agent = Agent(self.state_tracker.get_state_size(), constants)
        else:
            self.dqn_agent = DQNAgent(self.state_tracker.get_state_size(), constants)
        
    def _set_mode(self, mode=TRAIN):
        # Load goal File
        self.current_mode = mode
            
        if mode == TRAIN:
            current_path = self.train_file_path
        elif mode == DEV:
            current_path = self.dev_file_path
        else:
            current_path = self.test_file_path
                
        with open(current_path) as file:
            self.user_goals = list(map(json.loads, file.readlines()))

        if self.use_usersim:
            self.user = UserSimulator(self.user_goals, self.constants, self.database)

                
    def clear_logfile(self):
        if self.current_mode == TRAIN:
            current_path = self.log_train_path
        elif self.current_mode == TEST:
            current_path = self.log_test_path
        else:
            current_path = self.log_dev_path
        os.remove(current_path)
            
    def train(self):
        self._set_mode(mode=TRAIN)
        
    def dev(self):
        self._set_mode(mode=DEV)
        
    def test(self):
        self._set_mode(mode=TEST)
    