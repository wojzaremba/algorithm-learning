Learning Simple Algorithms from Examples
========================================

This is a framework to learn simple algorithms such as
copying, multi-digit addition and single digit multiplication
directly from examples. Our framework consists of a set of
interfaces, accessed by a controller. Typical
interfaces are 1-D tapes or 2-D grids that hold the input and output
data. 
The paper can be found at: http://arxiv.org/abs/1511.07275 . <br />
Moreover, the accompanying video https://www.youtube.com/watch?v=GVe6kfJnRAw gives a concise overview of our approach.


This software runs in Torch. Type <br />

`th main.lua`

to train the model for the addition task. 

The model generates traces of the intermediate solutions while training (in
directory ./movie/). They can be displayed by calling:

`python play.py`

