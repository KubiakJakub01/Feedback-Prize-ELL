# Feedback-Prize-ELL

## Overview

This repository is a code repository created for the Kaggle competition aimed at assessing the language proficiency of 8th-12th grade Feedback Prize English Language Learners ([ELLs](https://www.kaggle.com/competitions/feedback-prize-english-language-learning/overview)). The repository contains code and models developed by participants to improve automated feedback tools for ELLs using machine learning, natural language processing, and educational data analytics.

### Key features

- The repository contains a pipeline for training and evaluating models.
- Custom trainer with custom training loop written in PyTorch.
- Distributed training using PyTorch DistributedDataParallel.
- Wide range of parameters and artifacts like: pooling strategies, optimizers, schedulers, loss functions, etc.
- Tensorboard integration.
- Wandb integration.
- Dockerfile for building docker image with the required dependencies.
- One config file for training and evaluating models.

### Data

The dataset, known as the ELLIPSE corpus, is composed of argumentative essays written by English Language Learners (ELLs) in grades 8th to 12th. These essays have been scored based on six specific analytic measures: cohesion, syntax, vocabulary, phraseology, grammar, and conventions. These measures represent different aspects of essay writing proficiency, and higher scores indicate higher proficiency in each respective measure. The scores range from 1.0 to 5.0 in increments of 0.5.

### Goal

The goal of the competition is to predict the scores for each of the six measures for the essays in the test set. By developing models that can accurately predict these scores, participants aim to contribute to the improvement of automated feedback tools for ELLs.

## Repository Structure

The repository is structured as follows:

```
.
├── data
│   * Contains the data used in the competition
├── models
│   * Default directory for storing:
│       - Model checkpoints
│       - Model predictions
│       - Model evaluation results
│       - Model artifacts
│       - Tensorboard logs
├── notebooks
│   * Contains notebooks used for data exploration, model development, and model evaluation
├── src
│   * Contains source code with pipelines for training and evaluating models
├── wandb
│   * Contains wandb artifacts
├── .gitignore
├── README.md
├── requirements.txt
└── dockerfile
```

## Getting Started

To setup the repository you can use requirements.txt to install the required dependencies. It is recommended to use a virtual environment to install the dependencies. The following commands can be used to setup the repository:

```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Also you can use the dockerfile to build a docker image with the required dependencies. The following command can be used to build the docker image:

```
docker build -t feedback-prize-ell .
```

## Training and Evaluating Models

### Parameters

Before training and evaluating models, you need to specify the config yaml file that contains the parameters for the pipeline. You can follow the example in `config/sample_config.yaml` to create your own config file. For details list of parameters, please refer to `src/utils/params_parser.py`.

### Training and Evaluating Models

The repository contains a pipeline for training and evaluating models. The pipeline is implemented in the `src` directory. The pipeline can be run using the following command:

```
python src/train.py config/sample_config.yaml
```

Each run of the pipeline will create a new directory in the `models` directory. You also can track the runs using [wandb](https://wandb.ai/site). Tensorboard logs will be saved in the `models/model_name/logs` directory. You can use the following command to start tensorboard:
    
```
tensorboard --logdir models/model_name/logs
```

### Evaluations

To evaluate the models, you can use the `src/evaluate.py` script. The script take this same config file as training script. The following command can be used to evaluate the models:

```
python src/evaluate.py config/sample_config.yaml
```

Evaluation model architecture have to be the same as the model architecture used for training. The evaluation script will load the model from the checkpoint specified in the config file and evaluate the model on the given testset. The evaluation results will be saved in the specificed in the config `evaluation_dir` directory.


## My approach

My approach based on the use of the pretrained language models. I used the huggingface [transformers](https://huggingface.co/docs/transformers/index) framework to load BERT or any other transformers model. Then I used the pretrained model as an encoder and added a pooling layer followed by a linear layer for each of the six measures. I tried different pooling strategies: mean pooling, max pooling, and mean-max pooling. I also tried different learning rates and batch sizes. I used the AdamW optimizer with the default parameters. I used smooth_l1 loss function. Also I made same experiment with freezing the encoder and training only the top layers.

### Experiments

From the experiments I made, I found that the best results were obtained using the `bert-base-uncased` model with `LSTM` pooling. The best results were obtained with the following parameters:

- `learning_rate`: 2e-5
- `optimizer`: AdamW
- `scheduler`: cosine
- `loss_function`: smooth_l1
- `freeze_encoder`: True

Full list of parameters with tensorboard logs can be found in the `models/bert-base-uncased_2023-07-09-15:58:18` directory.

### Metrics

Metrics used in the competition is mcrmse (mean columnwise root mean squared error). The mcrmse is calculated as follows:

```
mcrmse = mean(sqrt(mean((y_true - y_pred)^2)))
```

For more details about the metrics, please refer to the [competition page](https://www.kaggle.com/competitions/feedback-prize-english-language-learning/overview/evaluation).

### Results

Best results obtained with the `bert-base-uncased` model with `LSTM` pooling and `freeze_encoder=True` are around `0.09` for the `mcrmse` metric. The results are obtained using the `bert-base-uncased_2023-07-09-15:58:18` model.

### Submission

TODO: Add submission details
