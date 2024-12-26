# FMCW_Doppler_PointTransformerNet
This repository reproduces [DopplerPTNet](https://iopscience.iop.org/article/10.1088/1742-6596/2809/1/012006). \
The codebase is an unofficial implementation of DopplerPTNet for semantic segmentation. \
This code is created based on the descroption of [DopplerPTNet](https://iopscience.iop.org/article/10.1088/1742-6596/2809/1/012006). If you find the implementation is not accurate, please raise an issue.

---
## Dependencies
- Ubuntu: 20.04 or higher
- PyTorch: 1.9.0 
- CUDA: 11.1 
- Hardware: Batch size 16 requires 48GB memory. 
- To create conda environment, command as follows:

  ```
  bash env_setup.sh dopplerpt
  ```

## Dataset preparation
- No public dataset for doppler lidar point cloud yet. 
- Download S3DIS [dataset](https://drive.google.com/uc?export=download&id=1KUxWagmEWnvMhEb4FRwq2Mj0aa3U3xUf) and symlink the paths to them as follows:

     ```
     mkdir -p dataset
     ln -s /path_to_s3dis_dataset dataset/s3dis
     ```

## Usage
- Shape classification on ModelNet40
  - For now, please use paconv-codebase branch.
- Part segmentation on ShapeNetPart
  - For now, please use paconv-codebase branch.
- Semantic segmantation on S3DIS Area 5
  - Train

    - Specify the gpu used in config and then do training:

      ```
      python train.py --config config/s3dis/s3dis_pointtransformer_repro.yaml
      ```

  - Test

    - Afer training, you can test the checkpoint as follows:

      ```
      CUDA_VISIBLE_DEVICES=0 python test.py config/s3dis/s3dis_pointtransformer_repro.yaml
      ```
  ---
## Experimental Results
---

## TODO
- [x] Semantic segmentation training and testing code
- [ ] Object detection training code
- [ ] Experiment result on non-FMCW LIDAR
- [ ] Experiment result on FMCW LIDAR
- [ ] CUDA Deployment code

---
## References

If you use this code, please cite:
```
@software{DopplerPointTransformerNet-pytorch,
  author = {Dingkang Wang},
  month = {12},
  title = {{DopplerPointTransformerNet-pytorch}},
  url = {https://github.com/dingkwang/DopplerPTNet-pytorch},
  version = {0.1},
  year = {2024}
}
```

## Acknowledgement
The description is from [DopplerPTNet: Object Detection Network with Doppler Velocity Information for FMCW LiDAR Point Cloud](https://iopscience.iop.org/article/10.1088/1742-6596/2809/1/012006).
I also refer [point-transformer](https://github.com/POSTECH-CVLab/point-transformer).
