# tasks-rule
A study of the role of tasks in shaping visual processing


# Tasks studied:
- Task 1 : Single-class classification task [link](https://github.com/hamzakeurti/tasks-rule/blob/master/data/single-class-classification-dataset.pkl).
  - Images are from MS-COCO.
  - There are 50,000 images in the training set and 10,000 images in the test set.
  - Pickle file organization structure = [training set, test_set], training set = [image id list, label list], test set = [image id list, label list].
  - image names: "./train2014/COCO_train2014_%012d.jpg" % (image_id)
- Task 2: Ten-class classification task
  - Images are from ImageNet.
  - The dataset consists of 60,000 images and ten categories. Each category contains 6,000 images.
  - The compressed file 10-class-classification-dataset.rar is in `/data4/chenhaoran` on img16.
  - There are ten folders in the compressed file and the name of each folder is the category name.
- Task 3: 20-class-multilabel classification task [link](https://github.com/hamzakeurti/tasks-rule/blob/master/data/20-multilabel-classification-task.pkl)
  - Images are from MS-COCO.
  - The dataset consists of 60, 000 images and twenty categories.
  - Pickle file organization structure = [dataset, label_description], dataset = [image id list, labels list], label description = {label: category name}.
  - image names: "./train2014/COCO_train2014_%012d.jpg" % (image_id) or "./val2014/COCO_val2014_%012d.jpg" % (image_id).

# fMRI data:
The [BOLD5000](https://bold5000.github.io/) dataset: fMRI responses of 4 subjects to 5254 images taken from COCO (2000 images), ImageNet (1916 images) and scene images (1000).

preprocessed images and fMRI data are now available in format imagefilename.pt : (image_array,fMRI_array), the fMRI array is the concatenation of the visual cortex voxels for subjects 1 to 3