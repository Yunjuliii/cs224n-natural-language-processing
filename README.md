# cs224n-natural-language-processing\
Here are my solutions for coding assignments of CS224N(Natural Language Processing) by Stanford 2021 winter. \
**Assignment2:**\
This assignment is about implementing word2vec algorithms. In the writing part, I figure out the loss function, gradient with respect to centerword vectors and outside vectors. To do this, basic linear algebra is needed, especially about the derivative of matrix. The coding part is mainly about making the writing part into practice. One thing important is that some matrix in coding is not in the same size with writing part, but transpose is needed. Another thing is that to make the programming more efficient, rather than use for loop to derive the answer, using numpy is recommended.\
The output of the picture is as follows:\
![image](https://github.com/Yunjuliii/cs224n-natural-language-processing/blob/main/a2/word_vectors.png)\
We can see that analogy is true from the pic that vector [king to queen] is very similar to vector [male to female]. Also, we successfully cluster the synonym words together, like the words about emotion and another bunch of adjective words like boring, amazing,etc.  


