# tasks-rule
A study of the role of tasks in shaping visual processing


# Tasks studied:
- Task 1 : Single-class classification task [link](https://github.com/hamzakeurti/tasks-rule/blob/master/data/single-class-classification-dataset.pkl).
  - Images are from MS-COCO.
  - There are 50,000 images in the training set and 10,000 images in the test set.
  - Pickle file organization structure = [training set, test_set], training set = [image id list, label list], test set = [image id list, label list].
- Task 2: Ten-class classification task
  - Images are from ImageNet.
  - The dataset consists of 60,000 images and ten categories. Each category contains 6,000 images.
  - The compressed file 10-class-classification-dataset.rar is in `/data4/chenhaoran` on img16.
  - There are ten folders in the compressed file and the name of each folder is the category name.

# fMRI data:
The [BOLD5000](https://bold5000.github.io/) dataset: fMRI responses of 4 subjects to 5254 images taken from COCO (2000 images), ImageNet (1916 images) and scene images (1000).
