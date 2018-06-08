# Leitner system 

Implementation of Leitner system for simultaneously training a given neural 
network and identifying spurious instances (those with wrong labels) 
in its input dataset. 

To use this code, you need to load your data (lines 182-191 in addition.py 
and 81-106 in reddit_twitter.py), design your favorite network architecture 
(lines 194-209 in addition.py and 109-117 in reddit_twitter.py), and set your
network parameters (lines 168-179 in addition.py and 73-79 in reddit_twitter.py). 

https://scholar.harvard.edu/hadi/spot
Please see this address for most recent update on spotting spurious data. 

# How to Use
python addition.py
or
python reddit_twitter.py

### Parameters 
**kern**: kernel function: this parameter must be set to 'lit' when fitting 
your model (line 215-216 in addition.py and 120-123 in reddit_twitter.py).
            
**acc_thr**: accuracy threshold: if network accuracy against an instance is greater 
than or equal to acc_thr, the spotter treats the instance as correctly classified; a 
smaller value than 1.0 can be used for more flexibility.



# Citation
Hadi Amiri, Timothy A. Miller, Guergana Savova. [Spotting Spurious Data with Neural Networks](https://scholar.harvard.edu/files/hadi/files/amiri_naacl18.pdf). NAACL 2018. 

# Contact
Hadi Amiri, hadi.amiri@childrens.harvard.edu

