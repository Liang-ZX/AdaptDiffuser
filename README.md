# AdaptDiffuser for KUKA

### [Project Page](https://adaptdiffuser.github.io/) | [Paper](https://arxiv.org/abs/2302.01877)

This is the official PyTorch implementation of paper:

> ### [AdaptDiffuser: Diffusion Models as Adaptive Self-evolving Planners](https://arxiv.org/abs/2302.01877)
> Zhixuan Liang, Yao Mu, Mingyu Ding, Fei Ni, Masayoshi Tomizuka, Ping Luo
> 
> ICML 2023 (Oral Presentation)

<img src="assets/teaser.png" width="55%">

## Preparation

### Environment
Code is tested on `CUDA 11.1` with `torch 1.9.1`

You should install [mujoco200](https://www.roboti.us/download.html) first and get your licence [here](https://www.roboti.us/license.html).
```shell
conda env create -f environment.yml
conda activate diffuser_kuka
pip install -e .
```

### Dataset
Download and extract the dataset for model pretraining from this [URL](https://www.dropbox.com/s/zofqvtkwpmp4v44/metainfo.tar.gz?dl=0) and save it as the following path 
```shell
${ROOT_DIR}/kuka_dataset
```

## Train the original KUKA stacking task (Seen Task)
To train the unconditional diffusion model on the block stacking task, you can use the following command:

```
python scripts/kuka.py
```

You may evaluate the diffusion model on unconditional stacking with

```
python scripts/unconditional_kuka_planning_eval.py
```

or conditional stacking with

```
python scripts/conditional_kuka_planning_eval.py
```

Samples and model checkpoints will be logged to `./results` periodically

## Guided Trajectory Generation and Selection
With the trained kuka model, you can generate KUKA Stacking data (bootstrapping) or KUKA Pick and Place data (adaptation).
### Generate KUKA Stacking data
```shell
python scripts/conditional_kuka_planning_eval.py --env_name 'multiple_cube_kuka_merge_conv_new_real2_128' --diffusion_epoch 650 --save_render --do_generate --suffix ${SAVE_DIR_NAME}

# You can specify the directory name to save data with --suffix
```

### Generate KUKA Pick and Place data
```shell
python scripts/pick_kuka_planning_eval.py --env_name 'multiple_cube_kuka_temporal_convnew_real2_128' --diffusion_epoch 650 --save_render --do_generate --suffix ${SAVE_DIR_NAME}

# You can specify the directory name to save data with --suffix
```

## Model Re-training
### Fine-tune the model with generated pick and place data (Unseen Task)
You should merge the generated dataset with the original kuka_dataset first and set it to `./pick2put_dataset/merge_1`
```shell
python scripts/kuka_fine.py --data_path "./pick2put_dataset/merge_1" --suffix merge1 --train_step 401000 --visualization

# set visualization True only when you can have a graphicX, unset visualization will not affect the result
```

## Evaluation


## Citation
If you find this code useful for your research, please use the following BibTeX entry.
```bibtex
@inproceedings{liang2023adaptdiffuser,
    title={AdaptDiffuser: Diffusion Models as Adaptive Self-evolving Planners},
    author={Zhixuan Liang and Yao Mu and Mingyu Ding and Fei Ni and Masayoshi Tomizuka and Ping Luo},
    booktitle = {International Conference on Machine Learning},
    year={2023},
}
```

## Acknowledgements

The diffusion model implementation is based on Michael Janner's [diffuser](https://github.com/jannerm/diffuser) repo.
The organization of this repo and remote launcher is based on the [trajectory-transformer](https://github.com/jannerm/trajectory-transformer) repo. 
We thank the authors for their great works! Also, extend my great thanks to members of [HKU-MMLAB](http://luoping.me/#about). 

