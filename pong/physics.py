import numpy as np
import yaml

with open("pong/config.yaml", 'r') as file:
    config = yaml.load(file, Loader=yaml.FullLoader)

inverse_x = np.array([[-1, 0],
                        [0, 1]])

inverse_y = np.array([[1, 0],
                      [0, -1]])

e_1 = np.array([1, 0])
e_2 = np.array([0, 1])

def get_rebonce_vector(d_center, vb):
    sgn_x = np.sign(vb[0])
    sgn_y = np.sign(vb[1])

    d = 2*d_center/config["player_lenght"]
    alpha = -sgn_x*sgn_y*np.arccos(np.dot(vb, sgn_x*e_1)/config["v_b_std"])
    theta = np.clip(config["normal_rebounce_coeff"]*alpha - config["odd_rebounce_coeff"]*sgn_x*np.arctan(d), -np.pi/3, np.pi/3)
    
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],
                                [np.sin(theta), np.cos(theta)]])
    return  -sgn_x*config["v_b_std"]*rotation_matrix @ e_1

def _move_player(p, v, dt):
    return np.clip(p + v*dt, config["player_lenght"]/2, 1-config["player_lenght"]/2)

def _move_ball(b, v, dt):
    return np.clip(b+v*dt, [0, 0], [config["lenght_win"], 1])

def _resolve_collision(p1, p2, b, vb):
    if b[0] == 0:
        if abs(b[1]-p1)<config["player_lenght"]/2:
            return True, get_rebonce_vector(b[1]-p1, vb)
        else:
            return False, vb
    elif b[0] == config["lenght_win"]:
        if abs(b[1]-p2)<config["player_lenght"]/2:
            return True, get_rebonce_vector(b[1]-p2, vb)
        else:
            return False, vb

    if b[1] == 1 or b[1] == 0:
        return True, inverse_y @ vb
    
    return True, vb
        