import requests
import json
import pandas as pd
from data_processing import find_path

# Function that gathers the player data based on their name
def get_data_by_name(name):
    hashmap = initialize_map()
    return gather_player_data(hashmap[name])


# Function to gather data for a specific player
def gather_player_data(player_id):
    base_url = f"https://fantasy.premierleague.com/api/element-summary/{player_id}/"
    response = requests.get(base_url)

    if response.status_code == 200:
        player_data = response.json()
        # Extract relevant data from player_data
        player_info = player_data['history']
        df = pd.DataFrame(player_info)
        return df
    else:
        print(f"Failed to retrieve data for the player. Status code: {response.status_code}")
        return None

# Function to gather data for all players for a specific gameweek
def gather_gameweek_data(gameweek):
    base_url = f"https://fantasy.premierleague.com/api/event/{gameweek}/live/"
    response = requests.get(base_url)

    if response.status_code == 200:
        gameweek_data = response.json()
        return gameweek_data
    else:
        print(f"Failed to retrieve data for gameweek {gameweek}. Status code: {response.status_code}")
        return None

# Function to initialize the mpa of names to ids
def initialize_map():

    df = pd.read_csv(find_path())

    # Select only 'player_id' and 'player_name' columns
    player_names_and_ids_df = df[['id', 'name']]
    nametoid = player_names_and_ids_df.set_index('name')['id'].to_dict()

    print(nametoid.keys())
    return nametoid

