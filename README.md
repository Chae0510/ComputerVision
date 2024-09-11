# ComputerVision_TeamProject
<br>

## Introduction
1. **Goal**: Improve fine-grained classification accuracy by modifying the model with our ideas
2. **Key Idea**
   
    a. Augmentation: By transforming the existing training data in various ways to virtually increase the size of the dataset, overfitting is prevented.
    <img width="861" alt="image" src="https://github.com/user-attachments/assets/7baef495-182f-4b5e-82f6-f2fea732c966">
    b. Resnet-D: Used a modified version of ResNet-50, known as ResNet-D, to train an image dataset that underwent an augmentation process, resulting in improved accuracy.
   <img width="483" alt="image" src="https://github.com/user-attachments/assets/8555835e-05c0-473b-9d05-e477fa75e5a0">

3. **Result**: 96.3%

## Dataset
#### CUB_200_2011_repackage_class50
The Caltech-UCSD Birds-200-2011 (CUB-200-2011) dataset is a widely utilized dataset for fine-grained visual categorization tasks, specifically designed for bird species recognition. It comprises 11,788 images across 200 bird subcategories. The dataset is split into 5,994 images for training and 5,794 for testing. Each image is meticulously annotated with details such as a subcategory label, 15 part locations, 312 binary attributes, and a bounding box. Additionally, it includes textual information in the form of ten single-sentence descriptions for each image, sourced through Amazon Mechanical Turk, with a requirement of at least 10 words without revealing subcategories and actions​​.[인용](https://paperswithcode.com/dataset/cub-200-2011)

## Method
### 📌 Data Augmentation
* Rotation: Rotates the image 30 degrees.
* Color: brightness is a value between 0.5 and 0.8, contrast is a value between 0 and 1, saturation is a value between 0.5 and 0.8, and hue is a value between 0 and 0.5. Random values ​​are continuously changed and applied.
* Horizontal_flip: Flips the image horizontally.
* Zoom: Fix the image to 448x448 size and then enlarge it.
* Crop: Randomly resize and crop to 448x448 size.
* Random: Apply one of the above techniques to the image randomly.

After applying various augmentation techniques to the original image, learning was performed by creating a combination of the original image and various techniques.

<img width="957" alt="image" src="https://github.com/user-attachments/assets/e34c99b7-e41e-4e43-82d2-b3ac7cafe903">

The summer-galaxy line in the chart above: In the first attempt, only one augmentation technique was applied to the original image and the model was trained only on this data, showing the lowest accuracy and highest loss value. 


#### After modification

<img width="514" alt="image" src="https://github.com/user-attachments/assets/d0a8acf9-945d-4f8f-a6f5-7ca46c45e011">


It was confirmed that the highest accuracy was output in these four cases.
However, in the case of the random color augmentation method used in number 1, if training was performed after fixing the seed value, accuracy was lowered, so it was not included in the next training.

#### Random Search
In each of the three cases, after going through the hyperparameter tuning process, through a random search process.

<img width="422" alt="image" src="https://github.com/user-attachments/assets/87d41adc-d8af-427e-8c12-0ebf6cd6a6a1">


Selects the best accuracy from each params set and outputs the params with the best accuracy among several accuracies. -> This params set must be applied to the test.

> (Hyperparameters were randomly selected and augmented data was tried 10 times.)



the parameter combination that yields the best accuracy value is batch size=32, learning rate=0.1, optimizer=sgd, epoch=20, and momentum=0 (default). I was able to confirm. 



### 📌 Model 

<img width="751" alt="image" src="https://github.com/user-attachments/assets/8bb9f32a-1cca-4e85-aef9-ae95c58ee6d7">


#### Using Resnet 50D

1. Avoid ignoring 3/4 of the input feature map
2. Change the structure of the first two conv blocks of path A in the downsampling block.

#### ResNet 50D vs. ResNet 18
<img width="786" alt="image" src="https://github.com/user-attachments/assets/e8084633-8f1b-489f-af56-a28cf65a9c0d">

**Layer Differences:**
- **ResNet18** has blocks like `BatchNorm2d`, `ReLU`, `BasicBlock`, `AdaptiveAvgPool2d`, and `Linear`.
- **ResNet50d** has more complex layers like `Bottleneck`, `AdaptiveAvgPool2d`, `Flatten`, and `SelectAdaptivePool2d`, which indicate a more sophisticated and deeper architecture.

**Key Differences:**
1. **Number of Parameters**: ResNet50d has more than twice the number of parameters compared to ResNet18.
2. **Memory Requirements**: ResNet50d consumes much more memory due to its deeper structure, especially during forward and backward passes, which makes it more computationally expensive.
3. **Model Size**: The total size of ResNet50d is significantly larger, indicating higher computational requirements for training and inference.


## Conclusion
#### Last Epoch & Test Accuracy

<img width="903" alt="image" src="https://github.com/user-attachments/assets/9bcace43-82a3-40b8-96cf-8f1b77808cb7">


#### Compare the grad cam results from our trained model with the original image and reference code.

> Original Image / Reference Code / Our Result

<img width="945" alt="image" src="https://github.com/user-attachments/assets/356599aa-5cfa-4884-86bd-9ad06f3513df">


After data augmentation, good model selection, and hyperparameter tuning, the model can be seen focusing on objects after zero focus on birds.


## Reference
- [resnet 변형](https://openaccess.thecvf.com/content_CVPR_2019/papers/He_Bag_of_Tricks_for_Image_Classification_with_Convolutional_Neural_Networks_CVPR_2019_paper.pdf)
- [위 논문 설명](https://bo-10000.tistory.com/133)
- [resnet](https://paperswithcode.com/model/resnet-d?variant=resnet34d)
- [resnet 구조](https://wjunsea.tistory.com/99)
- [resnet 18 설명](https://hnsuk.tistory.com/31)
- https://ropiens.tistory.com/32
- https://www.kaggle.com/code/a4885534/mlrc2022-re-hist
- https://dl.acm.org/doi/pdf/10.1145/3581783.3612165

