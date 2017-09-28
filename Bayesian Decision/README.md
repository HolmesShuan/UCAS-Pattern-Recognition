## Exercise
Using LDF and QDF to perform binary-classification on any two catagories of CIFAR-10 or MNIST or [HWDB](http://www.nlpr.ia.ac.cn/databases/handwriting/Download.html).
## Further Question
Using LDF and QDF to implement Multi-classification.
## Recommended Reference
http://scikit-learn.org/stable/modules/lda_qda.htm
## Performance
We evaluate QDF and LDF on Cifar-10. Similarly you may re-implement our code on MNIST.  
Class 1 (`6K` Training images): cat<br>
Class 2 (`6K` Training images): horse<br>
#### QDF : <br>
Each class contains `1K` test images.<br> 
accuracy@Cls1 = `93.1%`<br>
accuracy@Cls2 = `73.0%`<br>
average acc = `83.1%`<br>
#### LDF : <br>
accuracy@Cls1 = `58.4%`<br>
accuracy@Cls2 = `66.1%`<br>
average acc = `62.3%`<br>
#### Details : 
We first reduce RGB channels to luminance channel((in YCrCb color space), which has been proved that human eyes are most sensitive to that dimension. The same as MNIST, luminance was shown in the gray image. E.g.

![Ycbcr Y channel demo](https://github.com/HolmesShuan/UCAS-Pattern-Recognition/blob/master/Bayesian%20Decision/ConvertImageFromYCbCrToRGBExample_01%20(1).png)

Then, the feature dimensions of each classes have been reduced from **3072** to **1024**.  
Moreover, we implement PCA on feature matrix (**12000x1024**) to further reduce the correlation of different features.  
This operation can also solve the problem of singular covariance matrix.  
Finally, we perform QDF and LDF on pre-processed features.
#### Principal Components Selection
![LDF{:height="50%" width="50%"}](./figure_1.png)
![QDF{:height="50%" width="50%"}](./figure_2.png)


