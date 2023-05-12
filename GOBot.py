import json
import os

from tqdm import tqdm
import wandb
from _trainer import Trainer
from constants import DEV

class GOBot(Trainer):
    def test_run(self):
        """
        Runs the loop that tests the agent.

        Tests the agent on the goal-oriented chatbot task. Only for evaluating a trained agent. Terminates when the episode
        reaches NUM_EP_TEST.

        """

        print('Testing Started...')
        LOG_PATH = self.log_dev_path if self.current_mode == DEV else self.log_test_path
        if not os.path.isfile(LOG_PATH):
            with open(LOG_PATH, 'w') as file:
                pass 
        
        episode = 0
        period_success_total = 0
        period_reward_total = 0
        success_total = 0
        reward_total = 0
        wandb_success_total = 0
        wandb_reward_total = 0
        
        n = len(self.user_goals)
        progress_bar = tqdm(total=n)
        while episode < n:
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
                reward_total += reward
            
            period_success_total += success
            wandb_success_total += success
            success_total += success
            
            
            if episode % 10 == 0:
                wandb.log({"success_rate": wandb_success_total / 10, 'avg_reward': wandb_reward_total / 10})
                wandb_reward_total = 0
                wandb_success_total = 0
                
                with open(LOG_PATH, 'a') as file:
                    file.write(json.dumps({ 'goal': self.user.goal, 'history': self.state_tracker.history}, indent=4))
            
            if episode % self.train_freq == 0:
                # Check success rate
                success_rate = period_success_total / self.train_freq
                avg_reward = period_reward_total / self.train_freq
                
                print(f"Episode: {episode}, Success Rate: {success_rate}, Avg Reward: {avg_reward}")
    
                period_success_total = 0
                period_reward_total = 0
            
        print(f'Total:\nTotal Success: {success_total}, Total Success Rate: {success_total / n}, Total Avg Reward: {reward_total / n}')
        print('...Testing Ended')