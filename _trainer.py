import json
import os
from tqdm import tqdm
from _bot_structure import BotStructure
import wandb 


class Trainer(BotStructure):    
    def run_round(self, state, warmup=False):
        # 1) Agent takes action given state tracker's representation of dialogue (state)
        agent_action_index, agent_action = self.dqn_agent.get_action(state, use_rule=warmup)
        # 2) Update state tracker with the agent's action
        self.state_tracker.update_state_agent(agent_action)
        # 3) User takes action given agent action
        user_action, reward, done, success = self.user.step(agent_action)
        if not done:
            # 4) Infuse error into semantic frame level of user action
            self.emc.infuse_error(user_action)
        # 5) Update state tracker with user action
        self.state_tracker.update_state_user(user_action)
        # 6) Get next state and add experience
        next_state = self.state_tracker.get_state(done)
        self.dqn_agent.add_experience(state, agent_action_index, reward, next_state, done)

        return next_state, reward, done, success


    def warmup_run(self):
        """
        Runs the warmup stage of training which is used to fill the agents memory.

        The agent uses it's rule-based policy to make actions. The agent's memory is filled as this runs.
        Loop terminates when the size of the memory is equal to WARMUP_MEM or when the memory buffer is full.

        """

        print('Warmup Started...')
        total_step = 0
        progress_bar = tqdm(total=self.warmup_mem)
        while total_step < self.warmup_mem and not self.dqn_agent.is_memory_full():
            # Reset episode
            self.episode_reset()
            done = False
            # Get initial state from state tracker
            state = self.state_tracker.get_state()
            while not done:
                next_state, _, done, _ = self.run_round(state, warmup=True)
                total_step += 1
                state = next_state
                progress_bar.update(1)

        print('...Warmup Ended')


    def train_run(self):
        """
        Runs the loop that trains the agent.

        Trains the agent on the goal-oriented chatbot task. Training of the agent's neural network occurs every episode that
        TRAIN_FREQ is a multiple of. Terminates when the episode reaches NUM_EP_TRAIN.

        """

        print('Training Started...')
        
        LOG_PATH = self.log_train_path
        if not os.path.isfile(LOG_PATH):
            with open(LOG_PATH, 'w') as file:
                pass 
            
        episode = 0
        period_reward_total = 0
        period_success_total = 0
        wandb_reward_total = 0
        wandb_success_total = 0
        
        success_rate_best = 0.0
        
        progress_bar = tqdm(total=self.num_ep_train)
        while episode < self.num_ep_train:
            self.episode_reset()
            episode += 1
            progress_bar.update(1)
            done = False
            state = self.state_tracker.get_state()
            while not done:
                next_state, reward, done, success = self.run_round(state)
                state = next_state
                period_reward_total += reward
                wandb_reward_total += reward

            period_success_total += success
            wandb_success_total += success
            
            if episode % 10 == 0:
                wandb.log({"success_rate": wandb_success_total / 10, 'avg_reward': wandb_reward_total / 10, 'success_rate_best': success_rate_best})
                wandb_success_total = 0
                wandb_reward_total = 0
                
                with open(LOG_PATH, 'a') as file:
                    file.write(json.dumps({ 'goal': self.user.goal, 'history': self.state_tracker.history}, indent=4))

            # Train
            if episode % self.train_freq == 0:
                if self.early_stopping_rounds and (episode // self.train_freq > self.early_stopping_delay):
                    self.rounds_without_eval_improvements += 1
                # Check success rate
                success_rate = period_success_total / self.train_freq
                avg_reward = period_reward_total / self.train_freq
                # Flush
                if success_rate >= success_rate_best and success_rate >= self.success_rate_threeshold:
                    self.dqn_agent.empty_memory()
                # Update current best success rate
                if success_rate > success_rate_best:
                    print('Episode: {} NEW BEST SUCCESS RATE: {} Avg Reward: {}'.format(episode, success_rate, avg_reward))
                    success_rate_best = success_rate
                    self.dqn_agent.update_best_model()
                    self.dqn_agent.save_weights(save_best_model=True)
                    self.rounds_without_eval_improvements = 0
                
                if self.early_stopping_rounds and self.rounds_without_eval_improvements > self.early_stopping_rounds:
                    print(f"Early stopping occurred at episode: {episode}")
                    break 
                    
                period_success_total = 0
                period_reward_total = 0
                # Copy
                self.dqn_agent.copy()
                # Train
                self.dqn_agent.train()
        print('...Training Ended')

    def episode_reset(self):
        """
        Resets the episode/conversation in the warmup and training loops.

        Called in warmup and train to reset the state tracker, user and agent. Also get's the initial user action.

        """

        # First reset the state tracker
        self.state_tracker.reset()
        # Then pick an init user action
        user_action = self.user.reset()
        # Infuse with error
        self.emc.infuse_error(user_action)
        # And update state tracker
        self.state_tracker.update_state_user(user_action)
        # Finally, reset agent
        self.dqn_agent.reset() 
        