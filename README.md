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
  - 3 different region-of-interest were studied.

<br>
<div align="center"> 
  <img src="https://github.com/romaniegaa/Portfolio/blob/main/images/brain_areas.png" width="75%" height="75%">
</div>
<br>
    
  - The addition of batch normalization layers between MaxPooling and 2DConv layers was studied.
  - The addition of dropout layers after the batch normalization layers was studied.
  - Transfer learning from the VGG16 model was studied.
 
<h2 align="center">Results</h2>

### Architecture optimization

After testing easy architectures and not obtaining an evaluation accuracy higher than a 20% with overfitting and high validation loss, with the aim of understanding how the hyperparameters interact with the learning capability of the algorithm different combinations were tested. First, we evaluated the impact that the number of layers, number of filters and size of the kernel would have. Therefore, we trained 125 models with 1 to 5 layers, 16 to 256 number of filters and kernel sizes from 2 to 6. The validation accuracy showed values above 55% for lower values for filters (left figure). It was also observed that higher kernel sizes led to lower accuracy values. The loss did not show any straightforward conclusion across the axes (right figure).

<br>

<div align="center"> 
  <img src="https://github.com/romaniegaa/Portfolio/blob/main/images/graph1.png" width="75%" height="75%">
</div>

<br>

Afterwards, a new combination of hyperparameters was prepared. The kernel size was kept constant as (2 x 2). Therefore, number of layers, number of filters and dropout rate was investigated. These dropout layers were introduced after each MaxPooling layer and before the following 2D convolutional layer (or flatten layer for the last one). According to the scatterplots, higher validation accuracy of 59.71% was obtained with 3 layers, 32 filters and a dropout rate of 0.3 (left figure). At the same time the evaluation loss was 74.41% (right figure).

<br>

<div align="center"> 
  <img src="https://github.com/romaniegaa/Portfolio/blob/main/images/graph2.png" width="75%" height="75%">
</div>

<br>

Finally, by maintaining the dropout rate constant at 0.3, a dense layer was added before the output layer containing 24+n units, being n = 0, …, 4. As it can be observed in the 3D scatterplots, there was no obvious improvement in the accuracy of the model by inserting a dense layer before the output. However, best results were observed  then this layer had 128 units. Therefore, we employed the best conditions to date, containing three layers, with 32 filters, (2 x 2) kernel size and dropout layers with a dropout rate of 0.3, obtaining a prediction accuracy of 49.51% with a loss of 70.61%. 

<br>

<div align="center"> 
  <img src="https://github.com/romaniegaa/Portfolio/blob/main/images/graph3.png" width="75%" height="75%">
</div>

<br>

As we could see during the training, the algorithm was not capable of learning the classification task, as shown in the next figure:

<br>

<div align="center"> 
  <img src="https://github.com/romaniegaa/Portfolio/blob/main/images/graph4.png" width="75%" height="75%">
</div>

<br>

### Data agumentation
Data augmentation was performed for the three 2D slices, leading to 10 augmented images per original image. As we can see in the next figure, the learning capability of the algorithm improved because of the data augmentation. We changed to a kernel-size of 3 due to odd sizes being preferred due to symmetrically dividing the previous layer pixels around the output layer. However, the obtained prediction accuracy was low, a 53.84%; whereas the loss was 72.44%.

<br>

<div align="center"> 
  <img src="https://github.com/romaniegaa/Portfolio/blob/main/images/graph5.png" width="75%" height="75%">
</div>

<br>

Literature shows by region-of-interest-based volumetry that adults with ASD have reduced corpus-callosum, whereas surface-based morphometry studies show increased cortical thickness in the parietal lobes. Therefore, the same strategy was followed for the other two data slices. However, as shown in the following table, even though training curves looked better than before data augmentation, no good accuracy scores were obtained.

| IMAGE | Accuracy | Loss |
| ----- | ----- | ----- |
| A | 0.5384 | 0.7244 |
| B | 0.5433 | 0.6917 | 
| C | 0.4951 | 0.6952 |

### Batch normalization
In order to try to increase the accuracy of the algorithms two protocols were tested: first, batch normalization layers were inserted where the dropout layers belonged; second, both batch normalization layers and dropout layers were employed in this order. As we can see in the results table, prediction accuracies were not high overall for images B and C. However, we could observe in image A that both procedures yielded a higher accuracy.

| IMAGE | Dropout | Accuracy | Loss |
| --- | --- | --- | --- |
| A |   | 0.6490 | 2.8565 |
|   | + | 0.6779 | 0.7423 |
| B |   | 0.5529 | 4.2097 |
|   | + | 0.5288 | 1.0323 |
| C |   | 0.5385 | 2.3216 |
|   | + | 0.4519 | 0.7914 |

### Transfer learning
In order to test whether we could achieve a higher accuracy, transfer learning technique was tested on the augmented image A dataset and the pre-trained model VGG16.

<br>

<div align="center"> 
  <img src="https://github.com/romaniegaa/Portfolio/blob/main/images/graph6.png" width="75%" height="75%">
</div>

<br>

As we can see in the training curves, the model did indeed learn; however, when new data was presented, the algorithm yielded a low accuracy of 51.92% and a high loss of 378.60%.

<h2 align="center">Conclusion</h2>

Different algorithm architectures have been developed in order to achieve the correct classification of magnetic resonance images of patients with autism spectrum disorder and neurotypical ones. Among the procedures that have been carried out, the best method has been to use magnetic resonance imaging with a vertical slice showing both the corpus-callosum and the parietal lobe. In addition, the use of BatchNormalization layers followed by Dropout layers resulted in a model with an accuracy of 68% and a loss of 74%.

<br>

Although there is room for improvement, decent accuracy has been achieved with easily obtainable data. This type of data does not require any special reprocessing unlike those mentioned in the state of art. Furthermore, to the best of our knowledge, there is no precedent for direct employment of 2D brain MRI for classification of neurotypical brains and patients with autism spectrum disorder.

<h2 align="center">Used libraries</h2>

- ```asopse.words```: to convert ".svg" data to ".png" data.
- ```OpenCV```: to import the data.
- ```matplotlib```: to make 2D and 3D graphs.
- ```NumPy```: to manipulate the numeric data.
- ```os```: to iterate throught the paths.
- ```TensorFlow```: to build, train and evaluate the models.

