# DATA_DESCRIPTION: pathmnist

## What this is
PathMNIST, part of the MedMNIST v2 collection. Derived from a histopathology
study of colorectal cancer: 28x28 RGB image patches extracted from hematoxylin
and eosin (H&E) stained histology slides, downsampled from an original
resolution of 224x224. Source: NCT-CRC-HE-100K (train/validation split) and
CRC-VAL-HE-7K (test split, from a different clinical center; not loaded by this
project). Only the train split is used.

## Loading
file: pathmnist.npz
x_key: train_images
y_key: train_labels
image_shape: 28,28,3
url: https://zenodo.org/records/10519652/files/pathmnist.npz?download=1
md5: a8b06965200029087d5bd730944a56c1

## Structure
- 89,996 images (the "train" split only; the full MedMNIST collection also has
  10,004 "val" and 7,180 "test" images from a different clinic, neither loaded
  here).
- Each image is 28x28x3 (RGB), flattened to 2,352 features per sample.
- Feature values are raw pixel intensities, integers 0 to 255.
- Effectively dense; no meaningful sparsity, unlike scRNA-seq data.

## Labels
9 tissue-type classes, present for every image, coded 0-8 in this order:
adipose, background, debris, lymphocytes, mucus, smooth muscle, normal colon
mucosa, cancer-associated stroma, colorectal adenocarcinoma epithelium. Class
sizes range from 7,886 to 12,885 images (about 1.6x, mild imbalance, not
severe). These are pathologist-assigned tissue labels, not synthetic.

Labels may be used as a conditioning variable for class-conditional generation
or as a classifier target when the objective calls for it (e.g. class
balancing). Any evaluation must use held-out real images only; synthetic images
never enter an evaluation or test split.

## Known caveats
- The 28x28 downsampling from the original 224x224 patches is lossy;
  fine-grained histological texture is not preserved at this resolution.
- Pixel values are integers 0-255, so a "non-negative integer" check passes,
  but these are intensities, not counts.
- Staining color varies between slides; a generator can reproduce stain
  variation as well as tissue structure.
