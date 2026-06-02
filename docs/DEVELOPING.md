# MANUAL UI TESTING

## workflow 1

- upload images
- upload a masks
- train cellpose model
- download training dataset and model

## workflow 2

- upload images
- upload no masks
- upload cellpose model
- upload classifier model
- segment images with model
- classify masks with classifier
- download annotated dataset
- create 2 cell violin metrics plots
- download cell metrics dataset

## workflow 3

- upload images
- upload no masks
- upload no models
- draw masks
- classify masks into two groups
- download annotated images
- create 3 cell metrics violin plots
- download cell metrics dataset

## workflow 4

- upload images
- uploads masks
- download masks and images
- reupload downloaded masks and images in fresh session

## workflow 5

- upload images
- upload masks
- manually classify some of the masks into 3 groups
- train classifier
- download classifier training dataset
- use trained classifier to classifier all masks
- create a bar plot of mean areas for each group

## workflow 6

- upload images
- upload no masks
- upload no models
- draw boxes around cellpose
- predict masks with sam2
- manually classify masks into two groups
- download annotated images dataset
