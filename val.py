from GOBot import GOBot
import wandb 
import json 

with open('config.json') as file:
    config = json.load(file)

bot = GOBot(constants=config)

wandb.login()
run = wandb.init(
    # Set the project where this run will be logged
    project="go-bot-for-restaurant-reservation",
    # Track hyperparameters and run metadata
    config=config)

bot.dev()
bot.test_run()