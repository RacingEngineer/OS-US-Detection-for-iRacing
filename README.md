## Oversteer and understeer detection system for iRacing
This repository implements oversteer and understeer detection for [iRacing](https://www.iracing.com/) simulator. The detection system uses ANFIS model based on
[anfis_pytorch](https://github.com/jfpower/anfis-pytorch) implementation. The methodology is inspired by the work of 
Hirche, B. and Ayalew, B. [[1]](#1).

![demo_gif](readme_files/demo.gif)

## Data collection procedure

Data was collected on [Centripetal Circuit](https://www.iracing.com/tracks/centripetal-circuit/). During each test car was
running from south to north. Tests consisted of sine and dwell procedure with different velocity, maximum steering angle, 
dwell time and sine frequency. Dataset is balanced both for left and right turns.

## Datasets

Table below shows available datasets. Datasets are stored in data/datasets.

|     dataset_name      |    Vehicle     | Temperature | Cloud cover | Moisture  |    Wind     |  Humidity |
|:---------------------:|:--------------:|:-----------:|:-----------:|:---------:|:-----------:|:---------:|
| mx5_normal_conditions | Mazda MX-5 Cup |    79 F     |    Clear    |   None    | 1 mph North | 0% |

## Supported vehicles/models
Table below shows available models corresponding dataset name and vehicle. Models are stored in *data/models*.

| model_name |    Vehicle     |     dataset_name      |
|:----------:|:--------------:|:---------------------:|
| mx5_vars6  | Mazda MX-5 Cup | mx5_normal_conditions |


## Demo script
To run app turn on iRacing and use Python script *run_app.py*. Two bars will appear. Left one indicates amount of understeer
and the other one indicates oversteer. 
You can run inference without these bars with *real_time_inference.py*.


## Packages setup

```commandline
pip install -r requirements.txt
```

You have to install PyTorch on your own. Follow the instructions on [official](https://pytorch.org/) PyTorch site.

## Dependencies 
- PyTorch
- NumPy
- pandas
- matplotlib
- oclock==1.2.2
- pyirsdk==1.3.5

## Third-party software
- anfis_pytorch (MIT license)

## Author
Piotr Durawa \
email: durawa.p.soft@gmail.com

## References
<a id="1">[1]</a> 
Hirche, B. and Ayalew, B.,
"A Fuzzy Inference System for Understeer/Oversteer Detection Towards Model-Free Stability Control"
SAE Int. J. Passeng. Cars - Mech. Syst. 9(2):2016, doi:10.4271/2016-01-1630.