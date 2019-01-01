import os

graph_max_edge_number = 100000
save_progress_interval = 20000

# DATA_DIR = "D:\\iProject\\graphplot\\"
DATA_DIR = "D:\\iProject\\smalldata\\"
if not os.path.exists(DATA_DIR):
    os.mkdir(DATA_DIR)

TEMP_DIR = os.path.expanduser(os.path.join(DATA_DIR, 'temp'))
if not os.path.exists(TEMP_DIR):
    os.mkdir(TEMP_DIR)
