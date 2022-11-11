import matplotlib.pyplot as plt
import os
from datetime import datetime

FIG_NUM = 1
FIG_SAVE_LOC = f'./tmp_plots'
program = None

def show():
    global FIG_NUM
    assert program is not None, "Set the program name"
    date_time = datetime.now().strftime("%d%m%Y_%H%M%S")
    save_loc = f'{FIG_SAVE_LOC}/{date_time}/{program}'
    os.makedirs(save_loc, exist_ok=True)
    plt.savefig(f'{save_loc}/plot_{FIG_NUM}.png')
    FIG_NUM = FIG_NUM + 1

def visualize_set_program(p):
    global program
    program = p