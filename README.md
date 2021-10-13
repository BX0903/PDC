# Parameter Decoupling Strategy for Semi-supervised 3D Left Atrium Segmentation 

### Introduction

This repository is for our ICMV 2021 paper '[Parameter Decoupling Strategy for Semi-supervised 3D Left Atrium Segmentation](https://arxiv.org/abs/2109.09596)'. 

### Installation

This repository is based on PyTorch 1.7.0.

### Usage

1. Clone the repository:

   ```shell
   git clone https://github.com/BX0903/PDC.git
   cd PDC
   ```
2. Put the data in `data/2018LA_Seg_TrainingSet`.

3. Train the model:

   ```shell
   cd code
   python train_LA_dc.py
   ```

4. Test the model:

   ```shell
   python test_LA.py
   ```

5. Test the model which is reported in the paper:

   ```shell
   python test_LA.py --test_paper_model True
   ```

### Note for data

We provided the processed h5 data in the `data` folder. (Pre-processing data like existing work [UA-MT](https://github.com/yulequan/UA-MT))

You can refer the code in `code/dataloaders/la_heart_processing.py` to process your own data. 

### Citation

If PDC is useful for your research, please consider citing:

```
@misc{hao2021parameter,
      title={Parameter Decoupling Strategy for Semi-supervised 3D Left Atrium Segmentation}, 
      author={Xuanting Hao and Shengbo Gao and Lijie Sheng and Jicong Zhang},
      year={2021},
      eprint={2109.09596},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```



If you use the LA segmentation data, please also consider citing:

      @article{xiong2020global,
         title={A Global Benchmark of Algorithms for Segmenting Late Gadolinium-Enhanced Cardiac Magnetic Resonance Imaging},
         author={Xiong, Zhaohan and Xia, Qing and Hu, Zhiqiang and Huang, Ning and Vesal, Sulaiman and Ravikumar, Nishant and Maier, Andreas and Li, Caizi and Tong, Qianqian and Si, Weixin and others},
         journal={Medical Image Analysis},
         year={2020} }

### Questions

Please contact haoxuanting1997@163.com or gtabris@buaa.edu.cn

## Acknowledgement

- This code is adapted from [UA-MT](https://github.com/yulequan/UA-MT), [SASSNet](https://github.com/kleinzcy/SASSnet), [DTC](https://github.com/HiLab-git/DTC).
- We thank Dr. Lequan Yu, M.S. Shuailin Li and Dr. Xiangde Luo for their elegant and efficient code base.