import yaml

def get_config():
    with open("pong/config.yaml", 'r') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    return config