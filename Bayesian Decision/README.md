## Exercise
Using LDF and QDF to perform binary-classification on any two catagories of CIFAR-10 or MNIST or [HWDB](http://www.nlpr.ia.ac.cn/databases/handwriting/Download.html).
## Further Question
Using LDF and QDF to implement Multi-classification.
## Recommended Reference
http://scikit-learn.org/stable/modules/lda_qda.htm
## Performance
Class 1 (`6K` Training images): cat<br>
Class 2 (`6K` Training images): horse<br>
#### QDF : <br>
Each class contains `1K` test images.<br> 
accuracy@Cls1 = `93.1%`<br>
accuracy@Cls2 = `73.0%`<br>
average acc = `83.1%`<br>
#### LDF : <br>
#### Details : 
We first reduce RGB channels to luminance channel((in YCrCb color space), which has been proved that human eyes are most sensitive to that dimension. E.g.

