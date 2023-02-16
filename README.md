# MAP583-Projet_CLIP  
Implementing the method described in : https://arxiv.org/abs/2210.07183    

Classifier class:  
-needs a set of labels (depends on the dataset)  
-find descriptors (still to do)  
-can compute an appropriate threshold to decide whether a descriptor pertains to the image or not (still to do)  
-compute similarity for labels  
-find the best label  
-can explain why the classifier made a decision  


Issues:
-Threshold  
-As implemented now, a descriptor is likely to be considered not in the image if he isn't in the correct class, even if he does appear in the image  
