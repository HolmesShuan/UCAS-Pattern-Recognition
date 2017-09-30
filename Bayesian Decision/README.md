## Exercise
Using LDF and QDF to perform binary-classification on any two catagories of CIFAR-10 or MNIST or [HWDB](http://www.nlpr.ia.ac.cn/databases/handwriting/Download.html).
## Further Question
Using LDF and QDF to implement Multi-classification.
## Recommended Reference
http://scikit-learn.org/stable/modules/lda_qda.htm
## Performance
We evaluate QDF and LDF on Cifar-10. Similarly you may re-implement our code on MNIST.  
Class 1 (`4.5K` Training + `0.5K` Validation): cat<br>
Class 2 (`4.5K` Training + `0.5K` Validation): horse<br>
#### QDF : <br>
Each class contains `1K` test images.<br> 
accuracy@Cls1 = `68.2%`<br>
accuracy@Cls2 = `81.5%`<br>
average acc = `74.9%`<br>
#### LDF : <br>
accuracy@Cls1 = `56.6%`<br>
accuracy@Cls2 = `64.3%`<br>
average acc = `60.5%`<br>
#### Exp Details : 
We first reduce RGB channels to luminance channel((in YCrCb color space), which has been proved that human eyes are most sensitive to that dimension. The same as MNIST, luminance was shown in the gray image.   
E.g.  
![Ycbcr Y channel demo](./ConvertImageFromYCbCrToRGB.png)

Then, the feature dimensions of each classes have been reduced from **3072** to **1024**.  
Moreover, we implement PCA on feature matrix (**9000x1024**) to further reduce the correlation of different features.  
This operation presents a partial solution to singular covariance matrix.   
Finally, we perform QDF and LDF on pre-processed features.
#### Principal Components Selection
![LDF{:height="50%" width="50%"}](./figure1.png)
![QDF{:height="50%" width="50%"}](./figure2.png)

QDF becomes pathological when feature dimension is too large(over 300), which implies that covariance matrix is nearly singular.  
We find that ZCA might help to alleviate this problem. 
[Shrinkage Method](http://www.ledoit.net/honey.pdf) has better solved this problem.   
In short, add disturbance terms to the singular matrix.


