<h2 align="center">Classification of Autism in MRI using CNNs</h2>

<h2 align="center">Introduction</h2>

<br>

MRI has been widely employed for the detection of different brain related illnesses, such as: Autism Spectrum Disorder, Alzheimer and Parkinson. Deep learning has been widely used for this purpose; however, the classification performance is difficult to compare across studies.
<br>

<h2 align="center">Dataset</h2>

<br>

For this study, the dataset released by Yang et al. was used (<a href="https://arxiv.org/abs/2211.12421">1</a>). This dataset derives from the Autism Brain Imaging Data Exchange (ABIDE) where subjects are classified intro controls (healthy) and patients (suffering from autism). From 1025 MRIs, 537 correspond to healthy brains and 488 to patients. As reported by the authors, there are 873 male and 152 female participants. The dataset also includes 259 teenagers and 96 children. Among these MRIs, 312 have been obtained with the patient having their eyes closed, which has impact on which areas of the brain are activated.

<br>
  
The data is presented as in the following figure, where different 2D-slices of the same MRI 3D scan are shown:

<br>

![](https://github.com/romaniegaa/Portfolio/blob/main/images/brains.png)

<br>

Deshpande et al. determined that the motor cortex and the mid cingulate cortex are the most reproducible resting-state functional brain networks to separate autism and control groups (<a href="https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2017.00459/full">2</a>). Therefore, to pursue the objective images at coordinate z=0 are needed from every MRI. These are ares of interest:

<br>

<div align="center"> 
  <img src="https://images-provider.frontiersin.org/api/ipx/w=370&f=webp/https://www.frontiersin.org/files/Articles/251884/fnins-11-00459-HTML/image_m/fnins-11-00459-g008.jpg">
  </div>
###### ![](/images/brains.png)
<br>

<h2 align="center">Methodology</h2>

- **Data preprocessing:**
  - The images were converted from ".svg" to ".png" with the package ```aspose.words```.
  - They were loaded using the package ```cv2```.
  - The desired fragment was cropped and the RGB channels were normalized.
  - The labels were obtained and the training and testing dataset were produced.

- **Model building, training and evaluation:**
  - Easy architectures were first tested with both color and non-color images.
  - Hyperparameters were tunned:
    - Number of 2DConv and MaxPooling layers
    - Number of filters in the 2DConv layers
    - Kernel size in the 2DConv and MaxPooling layers
    - Dropout rate after MaxPooling layers
    - Addition of Dense layers and number of units before output
  - Data augmentation was performed to increase the amount of data by randomly performing flipping, cropping and zooming.
  - 3 different region-of-interest were studied
