# LSA
This is the implementation of Joint Local and Statistical Discriminant Learning via Feature Alignment  (LSA) published at Signal, Image and Video Processing (SIVP) 2019.
# Motivation
Image processing has attracted increasing attention in recent researches to solve domain shift problem where machine learning algorithms are applied to sets of unseen images. Domain shift problem occurs when the training (source domain) and test (target domain) sets are collected in different environmental conditions but in related domains. In this way, the adaptation across data distributions of the source and target datasets are suggested as domain adaptation framework to overcome the performance degradation. Joint Local and Statistical discriminant learning via feature Alignment (LSA), as a novel domain adaptation method, is proposed to find a cross-domain subspace by matching cross-domain distribution shift and adapting the class structures of the local and statistical distributions across the source and target domains, during the dimensionality reduction. Specifically, LSA projects samples into an embedded subspace which the distances across the samples from same class are minimized and the distances across samples from different classes are maximized, at each local and statistical area, during alignment of marginal and conditional distributions. Furthermore, the class densities of samples based on manifold structure in different classes are preserved to provide more separability across various classes. 
# Usage
The original code is written using Matlab R2016a. I think all versions after 2013 can run the code.

# Datasets
We test LSA on the most popular domain adaptation and transfer learning datasets: Office+Caltech256 (with SURF and decaf features), COIL20 and Digit (USPS and MNIST domains). The datasets is put into the [datasets](/datasets) folder.

# Demo
The basic demos to run on benchamark visual datasets are put into [demo](/demo) folder. Run demoOfficeCaltechSurf.m to run LSA on Office+Caltech256 with surf feature.
Run demoOfficeCaltechDecaf.m to run LSA on Office+Caltech256 with decaf feature.
Run demoCoil.m to run LSA on COIL20.
Run demoDigit.m to run LSA on Digit.

# Results
To evaluate the proposed method, comprehensive experiments have been conducted on benchmark cross-domain object and digit recognition datasets. Experimental results have verified the superiority of LSA with a large margin in average classification accuracy against several state-of-the-art distribution matching, discriminant learning and deep learning methods of domain adaptation. Moreover, the results have demonstrated the effectiveness of our proposed representation learning.


# Reference
Gholenji, E., Tahmoresnezhad, J. (2019). Joint Local and Statistical Discriminant Learning via Feature Alignment. Signal, Image and Video Processing. https://doi.org/10.1007/s11760-019-01587-1
