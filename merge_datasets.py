import os
import shutil

rootdir = 'dataset'
destination = 'dataset_merged'

# gather all files
for root, dirs, files in os.walk(rootdir):
    if len(files) > 0:
        act_dir = os.path.basename(root)
        print(act_dir)
        for file in files:
            old_name = os.path.join(root, file)

            # Initial new name
            new_name = os.path.join(destination, (act_dir + file))

            shutil.copy(old_name, new_name)
