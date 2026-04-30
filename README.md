# DAF_IRNet
Degradation-Aware Feature Disentanglement for All-in-One Image Restoration:

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19186882.svg)](https://doi.org/10.5281/zenodo.19186882)

- This code is directly related to the manuscript submitted to The Neurocomputing: `Degradation-Aware Feature Disentanglement for All-in-One Image Restoration.' 
If you use this code or models in your research, please cite the corresponding manuscript.
# Requirement
- Python 3.11
- Pytorch 2.0
- CUDA 11.7
- MATLAB R2023b
# Datasets
- AISTD (ISTD+) [link](https://github.com/cvlab-stonybrook/SID)
- LoL [link](https://www.kaggle.com/datasets/soumikrakshit/lol-dataset)
- BSD [link](https://drive.google.com/file/d/1BGwa1cdUordpbJpIjGT4uN_WZswesEDL/view?usp=drive_link)
- rain100L [link](https://drive.google.com/file/d/1f-Se88bwPE0rHCEnBpDRnuggoNg3mB1S/view?usp=drive_link)
# Pretrained models
The corresponding pretrained models:
- All-in-One and Individual Task [checkpoints](https://drive.google.com/file/d/1nWXJe-jwlU5xP1jd2oKGoZItTEu2WdYj/view?usp=drive_link)
# Test the model
You can directly test the performance of the pre-trained model as follows:
Modify the paths to dataset and pre-trained model. You need to modify the following path in the `test.py` or run
- python test.py --load [checkpoint number, e.g 690]
# Train
1. Download datasets and set the following structure

    ```
    -- AISTD_Dataset
       |-- train
       |   |-- train_A  # shadow image
       |   |-- train_B  # shadow mask (no use)
       |   |-- train_C  # shadow-free GT
       |
       |-- test
           |-- test_A  # shadow image
           |-- test_B  # shadow mask (no use)
           |-- test_C  # shadow-free GT

    -- LoL_Dataset
       |-- train
       |   |-- train_A  # low-light image
       |   |-- train_B  # low-light-free GT
       |
       |-- test
           |-- test_A  # low-light image
           |-- test_B  # low-light-free GT

    -- BSD_Dataset
       |-- BSDdataset
       |   |-- train  # denoise image
       |
       |-- test (BSD68)
           |-- noisy15  # noisy image
           |-- noisy25  # noisy image
           |-- noisy50  # noisy image
           |-- original  # clean image

    -- RAIN_Dataset
       |-- train
       |   |-- train_A  # rain image
       |   |-- train_B  # rain-free GT
       |
       |-- test
           |-- test_A  # rain image
           |-- test_B  # rain-free GT
# Evaluation
The results reported in the paper are calculated by the `matlab` script used in [previouse method](https://github.com/hhqweasd/G2R-ShadowNet/blob/main/evaluate.m)
# Testing results
The testing results on dataset  AISTD (ISTD+), LoL, BSD68, rain100L are:
- AISTD (ISTD+) [Results](https://drive.google.com/file/d/1INfkEPVWfYTqwFpoLIHP4oZdf4JBeHWl/view?usp=drive_link)
- LoL [Results](https://drive.google.com/file/d/1Fh1f1zn09q1QhVtzARItwFtMc27ZVt1D/view?usp=drive_link)
- BSD [Results](https://drive.google.com/file/d/1353Bk4kXMx_H8x1Ea8gqaWYIWoDJg3Ew/view?usp=drive_link)
- rain100L [Results](https://drive.google.com/file/d/1x0X9hCH8hpRJGG4bu4kEtQ8lh702TCqK/view?usp=drive_link)

# Contact
If you have any questions, please contact idreeskhan045@gmail.com/ huangying@cqupt.edu.cn
