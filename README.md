# Leitner system 

Implementation of Leitner system for simultaneously training a given neural 
network and identifying spurious instances (those with wrong labels) 
in its input dataset. 

To use this code, you need to load your data (lines 78-87 in addition.py), 
design your favorite network architecture (lines 90-105 in addition.py), and 
set your network parameters (lines 64-75 in addition.py). 

https://scholar.harvard.edu/hadi/spot
Please see this address for most recent updates on spotting spurious data. 

# How to Use
python addition.py


### Parameters 
**kern**: kernel function: this parameter must be set to 'lit' when fitting 
your model (line 108-112 in addition.py).
            
**acc_thr**: accuracy threshold: if network accuracy against an instance is greater 
than or equal to acc_thr, the spotter treats the instance as correctly classified; a 
smaller value than 1.0 can be used for more flexibility.



# Citation
Hadi Amiri, Timothy A. Miller, Guergana Savova. [Spotting Spurious Data with Neural Networks](http://aclweb.org/anthology/N18-1182). NAACL 2018. 

# Contact
Hadi Amiri, hadi.amiri@childrens.harvard.edu

