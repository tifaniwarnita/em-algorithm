# Expectation Maximization
Expectation maximization (EM) algorithm implementation using Python.

1. Assume that we have distributions come from two sets of data points, red and blue.
<img src="https://raw.githubusercontent.com/tifaniwarnita/em-algorithm/master/figures/00_distribution-known.png" width="400"> <img src="https://raw.githubusercontent.com/tifaniwarnita/em-algorithm/master/figures/01_distribution-unknown.png" width="400">

2. From those distribution, we can easily guess the correct estimation of the mean and covariance from each distribution.
<img src="https://raw.githubusercontent.com/tifaniwarnita/em-algorithm/master/figures/02_correct-estimation.png" width="400">

3. However, if we do not know the correct mean and covariance, we can start from guessing the mean and covariance such as this.
<img src="https://raw.githubusercontent.com/tifaniwarnita/em-algorithm/master/figures/03_first-guess.png" width="400">

4. Then, we can do the EM algorithm to find the correct numbers. For example, we do in the 10 iterations.
<img src="https://raw.githubusercontent.com/tifaniwarnita/em-algorithm/master/figures/04_itr-01.png" width="200"><img src="https://raw.githubusercontent.com/tifaniwarnita/em-algorithm/master/figures/04_itr-02.png" width="200"><img src="https://raw.githubusercontent.com/tifaniwarnita/em-algorithm/master/figures/04_itr-03.png" width="200"><img src="https://raw.githubusercontent.com/tifaniwarnita/em-algorithm/master/figures/04_itr-04.png" width="200"><img src="https://raw.githubusercontent.com/tifaniwarnita/em-algorithm/master/figures/04_itr-05.png" width="200"><img src="https://raw.githubusercontent.com/tifaniwarnita/em-algorithm/master/figures/04_itr-06.png" width="200"><img src="https://raw.githubusercontent.com/tifaniwarnita/em-algorithm/master/figures/04_itr-07.png" width="200"><img src="https://raw.githubusercontent.com/tifaniwarnita/em-algorithm/master/figures/04_itr-08.png" width="200"><img src="https://raw.githubusercontent.com/tifaniwarnita/em-algorithm/master/figures/04_itr-09.png" width="200"><img src="https://raw.githubusercontent.com/tifaniwarnita/em-algorithm/master/figures/04_itr-10.png" width="200">

5. After 10 iterations, we can get better numbers for guessing the parameter. We can see it by comparing to the correct numbers.

<img src="https://raw.githubusercontent.com/tifaniwarnita/em-algorithm/master/figures/05_final.png" width="400"><img src="https://raw.githubusercontent.com/tifaniwarnita/em-algorithm/master/figures/02_correct-estimation.png" width="400">