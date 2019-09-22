import os
import matplotlib.pyplot as plt

def save_fig(fig_id, CHAPTER_ID, PROJECT_ROOT_DIR = ".", \
             tight_layout = True, fig_extension = "png", resolution = 300):
    IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID)
    os.makedirs(IMAGES_PATH, exist_ok = True)
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format = fig_extension, dpi = resolution)