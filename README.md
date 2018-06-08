# Leitner system 

Implementation of Leitner system for simultaneously training a given neural 
network and identifying spurious instances (those with wrong labels) in its 
input dataset. 

### template code for text classification datasets
text_spotter_template.py provides a template code to identify spurious 
instances in textual datasets. To use this code, you need to load your dataset 
(lines 54-63), design your favorite network architecture (lines 97-105), and 
set your network parameters (lines 65-71). 

### running example based on addition dataset
addition.py loads data at lines 78-87, develops a sequence to sequence 
LSTM network for performing addition (lines 90-105). It outputs noisy instances 
that we injected into the datasets stored in 'data_addition/*' (values at the 
end of files indicate noise ratio, see paper for details).

https://scholar.harvard.edu/hadi/spot
Please see this address for most recent updates on spotting spurious data. 

# How to Use
python addition.py

or 

python text_spotter_template.py (after loading your data: lines 54-63)

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

