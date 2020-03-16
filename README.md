# ECE247-Final-Project
Repository of the final project of ECE247

By Che-Hsien Lin, Te-Yi Kan, Yan-Peng Chen, Yu-Ting Lai

In this project, we use U-Net and WGAN to do image segmentation.

# Dataset preparation
Before training, dataset must be downloaded by

`bash get_dataset.sh`

# Training
To train the U-Net model, please run

`bash train_unet.sh`

To train the WGAN model, please run

`bash train_wgan.sh`

# Testing
Download pretrained model, please run 

`bash get_pretrained_model.sh`

To test the performance, please run

`bash test.sh`

It will randomly pick 4 images to demonstrate the performance.