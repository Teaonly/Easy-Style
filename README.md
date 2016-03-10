# easyStyle
All kinds of neural style transformer.

This project collects many kinds of nerual style transformer , including

* dreamer mode

* gram matrix style 

* MRF style 

* guided style/patch transformer


My project is foccus on clean and simple implementation of all kinds of algorithm.


---

## 1. install and setup

Install following package of Torch7.

```
cunn
loadcaffe
cudnn
```

This project needs a GPU with 4G memory at least. Firstly you should download VGG19 caffe model.

```
cd cnn/vgg19
source ./download_models.sh
```

## 2. Quick demo

The .style files descript the arch of network used in style transformer, they are based on Lua language. 
The .style files is very simple ,all the paramters are configed in thease files.

### 2.1 dreamer mode mode

```
th ezstyle ./dreamer.style
```
<p>
<img src="/images/winter.png" height="256px" style="max-width:100%;">
<img src="/images/_results/dreamer.png" height="256px" style="max-width:100%;">
</p>



### 2.2 gram matrix mode 

```
th ezstyle ./gram.style
```
<p>
<img src="/images/trump.png" height="256px" style="max-width:100%;">
<img src="/images/picasso.png" height="256px" style="max-width:100%;">
<img src="/images/_results/gram.png" height="256px" style="max-width:100%;">
</p>


### 2.3 MRF mode

```
th ezstyle ./mrf.style
```
<p>
<img src="/images/ford.png" height="256px" style="max-width:100%;">
<img src="/images/lohan.png" height="256px" style="max-width:100%;">
<img src="/images/_results/mrf.png" height="256px" style="max-width:100%;">
</p>


### 2.4 guided mode

[WIP]

### 3. Resouces

All of code is coming from following projects, I have make them more simpler and stupid :).


https://github.com/chuanli11/CNNMRF

https://github.com/alexjc/neural-doodle

https://github.com/awentzonline/image-analogies

https://github.com/jcjohnson/neural-style



