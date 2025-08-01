from gr00t.experiment.data_config import DATA_CONFIG_MAP
from gr00t.utils.misc import any_describe
from gr00t.data.dataset import LeRobotSingleDataset
from gr00t.data.dataset import ModalityConfig
from gr00t.data.schema import EmbodimentTag

import os
import gr00t

# REPO_PATH is the path of the pip install gr00t repo and one level up
REPO_PATH = os.path.dirname(os.path.dirname(gr00t.__file__))
DATA_PATH = os.path.join(REPO_PATH, "/home/kleist/Documents/Database/test_0711a_test")
data_config = 'aloha_single_arm'

print("Loading dataset... from", DATA_PATH)

data_config_cls = DATA_CONFIG_MAP[data_config]
modality_configs = data_config_cls.modality_config()
transforms = data_config_cls.transform()

# 3. gr00t embodiment tag
embodiment_tag = EmbodimentTag.NEW_EMBODIMENT

# load the dataset
dataset = LeRobotSingleDataset(DATA_PATH, modality_configs, embodiment_tag=embodiment_tag,
                               video_backend='torchvision_av')

print('\n' * 2)
print("=" * 100)
print(f"{' Humanoid Dataset ':=^100}")
print("=" * 100)

# print the 7th data point
resp = dataset[7]
any_describe(resp)
print(resp.keys())
print("=" * 100)

# # show img
# import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt
#
# images_list = []
#
# for i in range(100):
#     if i % 10 == 0:
#         resp = dataset[i]
#         img = resp["video.middle_view"][0]
#         images_list.append(img)
#
#
# fig, axs = plt.subplots(2, 5, figsize=(20, 10))
# for i, ax in enumerate(axs.flat):
#     ax.imshow(images_list[i])
#     ax.axis("off")
#     ax.set_title(f"Image {i}")
# plt.tight_layout() # adjust the subplots to fit into the figure area.
# plt.savefig("test_load_dataset.png")


video_modality = modality_configs["video"]
state_modality = modality_configs["state"]
action_modality = modality_configs["action"]

dataset = LeRobotSingleDataset(
    DATA_PATH,
    modality_configs,
    transforms=transforms,
    embodiment_tag=embodiment_tag,
    video_backend='torchvision_av',
)

# print the 7th data point
resp = dataset[7]
any_describe(resp)
print(resp.keys())
print(resp["eagle_content"].keys())
print(resp["eagle_content"]['image_inputs'][0].save('0.png'))
print(resp["eagle_content"]['image_inputs'][1].save('1.png'))
print(resp["eagle_content"]['image_inputs'][2].save('2.png'))
