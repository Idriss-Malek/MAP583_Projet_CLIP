# MAP583-Projet_CLIP  
Implementing the method described in : https://arxiv.org/abs/2210.07183      

To use this repository, you can follow the tutorials.  
You can use the script to directly test this method on any dataset of images. the dataset should be given as an ImageFolder. First argument after script name is the root of the Image Folder and optional second argument is the location of the txt file with descriptors written with the json syntax.  
Don't forget to write your key in the config.ini file. Here is the syntax:

[openai]  
api_key = YOUR_API_KEY_HERE


**Classifier class:**  
-needs a set of labels (depends on the dataset)  
-find descriptors (still to test)  
-can compute an appropriate threshold to decide whether a descriptor pertains to the image or not (still to do)  
-compute similarity for labels  
-find the best label  
-can explain why the classifier made a decision  


**Issues:**    
-Threshold  
-As implemented now, a descriptor is likely to be considered not in the image if he isn't in the correct class, even if he does appear in the image  


**Solutions:**  
-For threshold, there is an attempt in the classifier class with the compute_threshold method   
-For explanations:  2 ways of computing the probabiltiies for descriptors:  
	+softmax on all descriptors (all labels combined)  
	+for each label, softmax on the set of descriptors  
	Second way seems to work better to explain for the cat dog example  
-With Clip, should we tokenize '{category} which is/has {descriptor}' or just '{descriptor}'. (paper says first solution and accroding to dog-cat example, it seems to wrok better).  

**Test on Caltech101 dataset**  
-Accuracy of the custom algorithm: 82.57583975791931%  
-Accuracy of the traditional zero shot prediction: 78.25313806533813%  
So we see that there is more than 4% improvement when using descriptors.  

**Test on Oxford flowers102 dataset**    
Accuracy of the custom algorithm: 64.67211842536926%  
Accuracy of the traditional zero shot prediction: 61.16741895675659%  
