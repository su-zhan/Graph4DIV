# Graph4DIV

Implementations of **Graph** **for** Search Result **Div**ersification (Graph4DIV) model descirbed in the paper:

"[Modeling Intent Graph for Search Result Diversification](https://dl.acm.org/doi/pdf/10.1145/3404835.3462872)"

Zhan Su, Zhicheng Dou, Yutao Zhu, Xubo Qin, and Ji-Rong Wen.

## Environment Requirement

The project is implemented using python 3.6 and tested in Linux environment. We use ``anaconda`` to manage our experiment environment. 

Our system environment and cuda version as follows:

```bash
Ubuntu 16.04.12
TITAN V CUDA Version: 10.1
```

Follow the steps to quickly build the environment ``Graph4DIV``:

```bash
conda create -n Graph4DIV python=3.6 pytorch torchvision cudatoolkit=10.1 cudnn=7.6.5 -c pytorch
pip install sklearn -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install pandas==1.1.3 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install transformers  -i https://pypi.tuna.tsinghua.edu.cn/simple
```

We add ``-i https://pypi.tuna.tsinghua.edu.cn/simple`` to acclerate the installation procedure, you could also remove it. 

## How To Reproduce Experimental Results

Since our experiement could be divided into two phases: **(1) Training for Classifier** **(2) Training for diversity ranking**, we introduce the utilization of these two processes, seperately.

1. **Training for diversity ranking.** If you would like to start your own training for diversity ranking model, build the dataset first by running the following commands sequently before starting your own training. In this section, we provide the intent coverage of documents given by the classifier so that you do not need to train the classifier.

   (1) Data Preprocess and Build Dataset.

   ```bash
   python run.py --mode data_preprocess
   python run.py --mode load_query
   python run.py --mode gen_train_data
   python run.py --mode build_graph
   ```

   (2) Start Training. 

   ```bash
   python run.py --mode train
   ```

2. **Training for Document Relation Classifier.** If you would like to train your own classifier, build the dataset first. To make our experiment credible and reproducible, we use a randomly shuffled sequence of query and load the sequence for dividing 5 fold datasets every time.

   (1) Get the ground-truth intent coverage from diversity judge. 

   ```bash
   python run.py --mode get_intent_cover
   ```

   (2) Get the document tokens for training.

   ```bash
   python run.py --mode get_doc_tokens
   ```

   (3) Divide 5 fold trianing and testing datasets for classifier.

   ```bash
   python run.py --mode divide_clf_dataset
   ```

   (4) Make dataset for every fold. (For example, make dataset for fold 1.)

   ```bash
   python run.py --mode make_dataset_clf --fold 1
   ```

   (5) Strart training for every fold.

   ```bash
   python Classifier.py --is_training True --fold 1
   ```

   (6) Testing for every fold.

   ```bash
   python Classifier.py --is_training False --fold 1
   ```

Due to the huge size of the classifier model files, we do not upload the models. In addition, the training for classifier with batch size of 16 requires 4 GPUs with at least 12GB memory. The required data files for training are avaliable at [dataset](http://playbigdata.ruc.edu.cn/dou/DSSA/data_cv.tar.gz).

## Reference

```
@inproceedings{10.1145/3404835.3462872,
author = {Su, Zhan and Dou, Zhicheng and Zhu, Yutao and Qin, Xubo and Wen, Ji-Rong},
title = {Modeling Intent Graph for Search Result Diversification},
year = {2021},
isbn = {9781450380379},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3404835.3462872},
doi = {10.1145/3404835.3462872},
booktitle = {Proceedings of the 44th International ACM SIGIR Conference on Research and Development in Information Retrieval},
pages = {736–746},
numpages = {11},
keywords = {search result diversification, graph neural network, intent graph},
location = {Virtual Event, Canada},
series = {SIGIR '21}
}
```

For any issues with the code, feel free to contact suzhan AT ruc.edu.cn
