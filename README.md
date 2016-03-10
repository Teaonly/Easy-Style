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

### 2.2 gram matrix mode 

```
th ezstyle ./gram.style
```

### 2.3 MRF mode

```
th ezstyle ./mrf.style
```

### 2.4 guided mode

WIP

### 3. Resouces

All of code is coming from following projects, I have make them more simpler and clean. 






