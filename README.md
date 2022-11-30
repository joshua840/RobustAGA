<div align="center">    
 
# Towards More Robust Interpretation via Local Gradient Alignment

<a href="https://www.python.org/"><img alt="Python" src="https://img.shields.io/badge/-Python 3.7+-blue?style=for-the-badge&logo=python&logoColor=white"></a>
<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/-PyTorch 1.9+-ee4c2c?style=for-the-badge&logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning 1.6+-792ee5?style=for-the-badge&logo=pytorchlightning&logoColor=white"></a>
<a href="https://black.readthedocs.io/en/stable/"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-black.svg?style=for-the-badge&labelColor=gray"></a>


[![Conference](https://img.shields.io/badge/AAAI-2023-4b44ce.svg)](https://aaai.org/Conferences/AAAI-23/)
</div>



## 📌&nbsp;&nbsp;Introduction

> It is required to have prior knowledge of [PyTorch](https://pytorch.org) and [PyTorch Lightning](https://www.pytorchlightning.ai). Also, we recommend you to use at least one of the logging framework [Weights&Biases](https://wandb.com) or [Neptune](https://neptune.ai).


---
 
## 1. Description
This repository is an official implementation for [Towards More Robust Interpretation via Local Gradient Alignment](https://arxiv.org/abs/2211.15900). 
- Supported training losses: CE loss, Hessian regularizer, and l2+cosd regularizer 
- Supported dataset: CIFAR10 and ImageNet100
- Supported model: `LeNet` and `ResNet18`

## 2. How to install
We highly recommend you to use our conda environment.
```bash
# clone project   
git clone https://github.com/joshua840/RobustAttributionGradAlignment.git

# install project   
cd RobustAGA
conda env create -f environment.yml 
conda activate agpu_env
 ```   

## 3. Project Structure

<!-- ├── example                 <- Will be integraged with notebooks dir
│
├── notebooks               <- Useful jupyter notebook examples are given
│ -->

<br>

Our directory structure looks like this:

```
├── project                 
│   ├── module                             <- Every modules are given in this directory
│   │   ├── lrp_module                     <- Modules to get LRP XAI are in this directory
│   │   ├── models                         <- Models
│   │   ├── utils                          <- utilized
│   │   ├── pl_classifier.py               <- basic classifier
│   │   ├── pl_hessian_classifier.py       <- hessian regularization
│   │   ├── pl_l2_plus_cosd_classifier.py  <- l2 + cosd regularization
│   │   ├── test_adv_insertion.py          <- run Adv-Insertion test
│   │   ├── test_insertion.py              <- run Insertion test
│   │   ├── test_rps.py                    <- run Random Perturbation Similarity test
│   │   ├── test_taps_saps.py              <- run adversarial attack test
│   │   └── test_upper_bouond.py           <- run upper bound test
│   │
│   ├── main.py                            <- run train & test
│   └── test_main.py                       <- run advanced test codes
│ 
├── scripts                                <- Shell scripts
│
├── .gitignore                             <- List of files/folders ignored by git
├── environment.yml                        <- anaconda environment
└── README.md
```

<br>
 
## 4. Train model
You can check the arguments list by using -h
### 4.1 Arguments for CE-loss trainer

 ```bash
python project/main.py -h
```

```
usage: main.py [-h] [--seed SEED] [--regularizer REGULARIZER] [--loggername LOGGERNAME] [--project PROJECT] [--dataset DATASET] [--model MODEL]
               [--activation_fn ACTIVATION_FN] [--softplus_beta SOFTPLUS_BETA] [--optimizer OPTIMIZER] [--weight_decay WEIGHT_DECAY]
               [--learning_rate LEARNING_RATE] [--milestones MILESTONES [MILESTONES ...]] [--num_workers NUM_WORKERS] [--batch_size_train BATCH_SIZE_TRAIN]
               [--batch_size_test BATCH_SIZE_TEST] [--data_dir DATA_DIR]

optional arguments:
  -h, --help            show this help message and exit
  --seed SEED           random seeds (default: 1234)
  --regularizer REGULARIZER
                        A regularizer to be used (default: none)
  --loggername LOGGERNAME
                        a name of logger to be used (default: default)
  --project PROJECT     a name of project to be used (default: default)
  --dataset DATASET     dataset to be loaded (default: cifar10)

Default classifier:
  --model MODEL         which model to be used (default: none)
  --activation_fn ACTIVATION_FN
                        activation function of model (default: relu)
  --softplus_beta SOFTPLUS_BETA
                        beta of softplus (default: 20.0)
  --optimizer OPTIMIZER
                        optimizer name (default: adam)
  --weight_decay WEIGHT_DECAY
                        weight decay for optimizer (default: 4e-05)
  --learning_rate LEARNING_RATE
                        learning rate for optimizer (default: 0.001)
  --milestones MILESTONES [MILESTONES ...]
                        lr scheduler (default: [100, 150])

Data arguments:
  --num_workers NUM_WORKERS
                        number of workers (default: 4)
  --batch_size_train BATCH_SIZE_TRAIN
                        batchsize of data loaders (default: 128)
  --batch_size_test BATCH_SIZE_TEST
                        batchsize of data loaders (default: 100)
  --data_dir DATA_DIR   directory of cifar10 dataset (default: /data/cifar10)
```

### 4.2 Arguments for trainer with regularizer
Setting --regularizer option will print out some additional arguments
 ```
python project/main.py --regularizer l2_cosd -h

l2_cosd arguments:
  --eps EPS
  --lamb LAMB
  --lamb_c LAMB_C
  --detach_source_grad DETACH_SOURCE_GRAD

python project/main.py --regularizer hessian -h

Hessian arguments:
  --lamb LAMB

```

### 4.3 Hidden arguments
pytorch_lightning offers useful arguments for training. For example, we used `--max_epochs` and `--default_root_dir` in our experiments. We recommend the user to refer to the following link to check the argument lists.

(https://pytorch-lightning.readthedocs.io/en/latest/api/pytorch_lightning.trainer.trainer.Trainer.html?highlight=trainer)

## 5. Loggers
We offer three options of loggers.
- Tensorboard (https://www.tensorflow.org/tensorboard)
   - Log & model checkpoints are saved in `--default_root_dir`
   - Logging test code with Tensorboard is not available.
- Weight & bias (https://wandb.ai/site)
   - Generate a new project on the WandB website. 
   - Specify the project argument `--project` 
- Neptune AI (https://neptune.ai/)
   - Generate a new project on the neptune website.
   - export NEPTUNE_API_TOKEN="YOUR API TOKEN"
   - export NEPTUNE_ID="YOUR ID"
   - Set `--default_root_dir` as `output/YOUR_PROJECT_NAME`



## 6. test model
Likewise, You can check the options for test code. 
 ```bash
python project/test_main.py --test_method aopc -h
python project/test_main.py --test_method adv -h
python project/test_main.py --test_method adv_aopc -h
python project/test_main.py --test_method rps -h
python project/test_main.py --test_method upper_bound -h
```

For those above test codes, you should specify the `--exp_id` argument. You can check the exp-id in your web project page and it seems like `EXP1-1` for Neptune and `1skdq34` for WandB. Above runs will append the additional logs in to your projects.

## 7. Import lightning modules

This project is setup as a package which means you can now easily import any file into any other file like so:
```python
from project.pl_classifier import LitClassifier
from project.module.utils.data_module import CIFAR10DataModule
from project.module.utils.data_module import ImageNet100DataModule

# Data
data_module = CIFAR10DataModule()
data_module = ImageNet100DataModule()

# Model
model = LitClassifier(model=model_name, activation_fn=activation_fn, softplus_beta=beta).cuda()


# train
trainer = Trainer()
trainer.fit(model, data_module)

# test using the best model!
trainer.test(model, data_module)
```


## 8. Import lightning module on jupyter notebook

This project is setup as a package which means you can now easily import any file into any other file like so:
```python
from project.test_upper_bound import LitClassifierUpperBoundTester as LitClassifier
from project.module.utils.data_module import CIFAR10DataModule, ImageNet100DataModule
from project.module.utils.interpreter import Interpreter

# Data
data_module = CIFAR10DataModule(dataset='cifar10',batch_size_test=10,data_dir = '../data/cifar10')
data_module.prepare_data()
data_module.setup()
test_loader = data_module.test_dataloader()

x_batch, y_batch = next(iter(test_loader))
x_s = x_batch.cuda().requires_grad_()
y_s = y_batch.cuda()

# Model
ckpt_path = f'YOUR_CHECKPOINT_PATH'
ckpt = torch.load(model_path)
args = ckpt['hyper_parameters']
model = LitClassifier(**args).cuda()
model.load_state_dict(ckpt['state_dict'])
model.eval()

# Use interpreter
yhat_s = model(x_s)
h_s = Interpreter(model).get_heatmap(x_s, y_s, yhat_s, "grad", 'standard', 'abs', False).detach()
```

### Citation   
```
TBU
@article{YourName,
  title={Your Title},
  author={Your team},
  journal={Location},
  year={Year}
}
```   
