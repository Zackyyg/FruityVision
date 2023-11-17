<h1>Fruit Classification Model</h1>

<h2>Overview</h2>
This project uses a Convolutional Neural Network built with TensorFlow and Keras to classify images of fruits. The classifier can distinguish between six different categories: fresh apples, fresh bananas, fresh oranges, rotten apples, rotten bananas, and rotten oranges.


<h3>instal dependencies</h3>

```bash
pip install tensorflow keras numpy pillow
```
<h3>Dataset Preparation</h3>

1. Download dataset from [Kaggle](https://www.kaggle.com/datasets/sriramr/apples-bananas-oranges/data)

2. extract dataset in project folder

3. remove duplicate dataset
```bash
rm -r PyVision/original_data_set/original_data_set
```

<h3>Training the model</h3>

```bash
python train.py
```

<h3>Make Predictions</h3>

```bash
python predict.py images/banana.jpg
```

python predict.py images/orange.jpg
