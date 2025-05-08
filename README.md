# BPNet

BPNet: Bézier Primitive Segmentation on 3D Point Clouds  ([IJCAI-23](https://www.ijcai.org/proceedings/2023/84) & [TCSVT-24](https://ieeexplore.ieee.org/document/10789135). The journal version is also available on [here](https://drive.google.com/file/d/1Yv5qtdb4o9ka1wXNt5CgKwxuDAFVfmOj/view?usp=sharing).

<div align="center">
  <img width="100%" alt="BPNet Pipeline" src="teaser/pipeline.png">
</div>


### Data Preparation
Please download the pre-processed [ABC dataset](https://drive.google.com/file/d/15u9hpQqurYhzNIZrnCVejCoAYXmr_U8-/view?usp=sharing).

unzip the dataset and put it in the data folder, or modify the data root path in the `options.py`

### Training
configure your training settings in `options.py`, and then:

```python train.py```

### Testing
configure your testing settings in `options.py`, and then:

```python test.py```


