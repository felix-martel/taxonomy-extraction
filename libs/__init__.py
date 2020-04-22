import json

with open("libs/config.json", "r") as conf:
    CONFIG = json.load(conf)
CONST = CONFIG["constants"]