import os
import shutil

source_dir = './dogs-vs-cats-redux-kernels-edition/train'
target_dir = './dogs-vs-cats-redux-kernels-edition/train_sorted'

os.makedirs(os.path.join(target_dir, 'cat'), exist_ok=True)
os.makedirs(os.path.join(target_dir, 'dog'), exist_ok=True)

for filename in os.listdir(source_dir):
    if filename.startswith('cat'):
        shutil.copy(
            os.path.join(source_dir, filename),
            os.path.join(target_dir, 'cat', filename)
        )
    elif filename.startswith('dog'):
        shutil.copy(
            os.path.join(source_dir, filename),
            os.path.join(target_dir, 'dog', filename)
        )
