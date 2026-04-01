### Requirements

The project is tested with the following environment:

- python == 3.7.15  
- pytorch == 1.12.1  
- torchmetrics == 0.10.3  
- scikit-learn == 1.0.2  
- scipy == 1.7.3  

You can install dependencies via:

```bash
pip install torch==1.12.1 torchmetrics==0.10.3 scikit-learn==1.0.2 scipy==1.7.3
## Dataset Preparation

Download the source and target datasets from:

https://pan.baidu.com/s/1BY0EqAWe1BOherY7kZypHQ?pwd=vkde

After downloading, extract the files and place them into the `datasets/` directory.

### Directory Structure

```bash
datasets/
├── Pavia_7gt
├── PaviaC
├── Houston_7gt
├── Houston18
├── HyRANK
└── Yancheng
