## Exercise
Given 5K smaples obey unknown gaussian mixture distribution (in the form of `a1*N(mu_1,Sigma_1)+a2*N(mu_2,Sigma_2)`),  
using [obversation vectors](https://github.com/HolmesShuan/UCAS-Pattern-Recognition/tree/master/Expectation-Maximization/data) to estimate a1, a2, mu_1, mu_2, Sigma_1 and Sigma_2. 

## Theorem & Proof
<img src="./proof1.png" width="612" height="707" />
<img src="./proof2.png" width="500" height="630" />

## Experiment Results
<img src="./img/Original.bmp" width="400" height="300" />

#### Observed feature distribution
<img src="./img/iter30_1.bmp" width="400" height="300" />

#### Estimated distribution (initialization : mu_1=[0 7]; mu_2=[7 0]; Sigma_1 = Sigma_2 = I)
<img src="./img/iter30_2.bmp" width="400" height="300" />

#### Estimated distribution (initialization : mu_1=[2 3]; mu_2=[5 1]; Sigma_1 = Sigma_2 = I)
<img src="./img/iter30_3.bmp" width="400" height="300" />

#### Estimated distribution (initialization : mu_1=[2 3]; mu_2=[5 1]; Sigma_1 = Sigma_2 = 3\*I)
### Estimated Parameters :
##### Mu_1 = [0 7] Sigma_1 = [2.15 -0.125; -0.125 2.90]
##### Mu_2 = [7 0] Sigma_2 = [3.90  0.161;  0.161 1.06]
##### alpha_1 = 0.4; alpha_2 = 0.6
