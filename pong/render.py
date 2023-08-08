import pygame
import yaml

win_size = 500

with open("pong/config.yaml", 'r') as file:
    config = yaml.load(file, Loader=yaml.FullLoader)


pygame_objects = {"screen" : None, "clock" : None}

def transform_ball_pos(pos, size):
    return config["player_width"]/2 + pos*(size-config["player_width"]/config["lenght_win"])


def render(env):
    if pygame_objects["screen"] is None:
        pygame_objects["screen"] = pygame.display.set_mode((win_size*config["lenght_win"], win_size))
        pygame_objects["clock"] = pygame.time.Clock()
    
    screen, clock = pygame_objects["screen"], pygame_objects["clock"]

    game = env.game

    screen.fill((0, 0, 0))

    size = screen.get_height()
    player_length = game.get_player_length()
    r = size/60

    rect1 = pygame.Rect(0, 0, config["player_width"], player_length*size-r)
    rect1.center = (0, game.p1*size)
    pygame.draw.rect(screen, (255, 255, 255), rect1)

    rect2 = pygame.Rect(0, 0, config["player_width"], player_length*size-r)
    rect2.center = (size*config["lenght_win"], game.p2*size)
    pygame.draw.rect(screen, (255, 255, 255), rect2)

    pygame.draw.circle(screen, (255, 255, 255), transform_ball_pos(game.b, size), r)

    run = True
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run = False
    
    pygame.display.update()

    dt = clock.tick(60)/1000/env.skip

    return dt, run