# About project

This project realizes Hough Transform algorithm for detecting planes in 3D space.
It works with point clouds, 
which can be represented as arrays of 3D points with shape (?, 3)

The closest analogy of this algorithm
is cv2.HoughLines() function from [OpenCV](https://en.wikipedia.org/wiki/OpenCV) library
that detects 2d lines in image.

Algorithm is written in python3, so it can have some problems with performance. 
I have plans to rewrite it in C++ some day, but these plans are vague, haha :)

# About Hough Transform

[Hough transform](https://en.wikipedia.org/wiki/Hough_transform) 
is a method of finding different spatial patterns. 
Theoretically, it can be applied to any shape.

It transforms points into parameter space where 
locations of similar patterns coinside.

To do so, we need to allocate accumulator tensor with 
all possible values in our parameter space and then for each point 
add one to some positions in accumulator.

After finishing this procedure for all points, 
we then can find extrema in parameter space.
Thus we can find sets of parameters that are 
shared by maximal amount of points.

# Pros and cons

++ Unlike other algorithms like RANSAC, 
Hough Transform can find multiple patterns simultaneously.

++ Algorithm can be significantly sped up by restricting the parameter space
by it's size or accuracy.

+- Algorithm has a lot of hyperparameters and is very sensitive to them. 
However, it's quite easy to adjust these parameters 
for your application if you know math behind each of them.

-- If angle variety crosses 0 or 180 degress (for example, -10°..+10°), 
the accumulator will be split into two. 
It can be fixed programmatically, though it's not fixed in current realization.
To overcome that, you'd better rotate your point cloud 
so that expected planes parameters won't exceed 0°-180° borders.

# Installation

Setup python3 virtual environment:

* Install python3.6

```bash
sudo apt-get update

sudo apt-get install software-properties-common

sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt-get update

sudo apt-get install python3-pip (not sure if this nessesary)
sudo apt-get install python3.6
sudo apt-get install python3.6-venv
```

* Create virtual enviroment:
```bash
python3.6 -m venv venv
```
* Activate environment
```bash
source venv/bin/activate
```
* Install requirements:
```bash
venv/bin/pip install --upgrade pip && venv/bin/pip install -r requirements.txt
```

# Usage

See jupyter notebook for use cases.

```bash
venv/bin/jupyter notebook
```

It's also recommended to read docstring for the function attentively.