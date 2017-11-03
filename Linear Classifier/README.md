## Exercise
1. Write a program to implement the "batch perception" algorithm.    
(1) Starting with a = 0, apply your program to the training data from w_1 and w_2. 
Note that the number of iterations required for convergence.    
(2) Apply your program to the training data from w_3 and w_4. 
Note that the number of iterations required for convergence.  
2. Implement the Ho-Kashyap algorithm and apply it to the training data from w_1 and w_3.
Repeat to apply it to the training data from w_2 and w_4. Point out the training errors, and give some analyses.    
3. By using MSE multi-class extension to construct a classifier, learn from samples [0:8] to test samples[9:10]. 

## Data Distribution
<img src="./img/w1_w2.bmp" width="300" height="200" />
<img src="./img/w3_w2_1.bmp" width="300" height="200" />

## Experiemnts
### Batch Update
<img src="./img/w1_w2_2.bmp" width="400" height="300" />
<img src="./img/w3_w2_2.bmp" width="400" height="300" />

### Widrow-Hoff
<img src="./img/w1_w3.bmp" width="400" height="300" />
<img src="./img/w4_w2_2.bmp" width="400" height="300" />

### Ho-Kashyap
#### The linearly non-separable problem will converge to e<0, but linearly separable problem will converge to zeor with b>0.
<img src="./img/w1_w3_2.bmp" width="400" height="300" />
<img src="./img/w1_w3_3.bmp" width="400" height="300" />
<img src="./img/w4_w2.bmp" width="400" height="300" />
<img src="./img/w4_w2_1.bmp" width="400" height="300" />

### Kelser Constructor
<img src="./img/acc.bmp" width="400" height="300" />

Train Accuracy | Test Accuracy
------------ | -------------
88.54 | 87.50
