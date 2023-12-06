# cs224n-natural-language-processing
Here are my solutions for coding assignments of CS224N(Natural Language Processing) by Stanford 2021 winter. \
**Assignment2:**\
This assignment is about implementing word2vec algorithms. In the writing part, I figure out the loss function, gradient with respect to centerword vectors and outside vectors. To do this, basic linear algebra is needed, especially about the derivative of matrix. The coding part is mainly about making the writing part into practice. One thing important is that some matrix in coding is not in the same size with writing part, but transpose is needed. Another thing is that to make the programming more efficient, rather than use for loop to derive the answer, using numpy is recommended.\
The output of the picture is as follows:\
![image](https://github.com/Yunjuliii/cs224n-natural-language-processing/blob/main/a2/word_vectors.png)\
We can see that analogy is true from the pic that vector [king to queen] is very similar to vector [male to female]. Also, we successfully cluster the synonym words together, like the words about emotion and another bunch of adjective words like boring, amazing,etc.  
**Assignment3**\
The wirrting part of the assignment introduced 2 tricks, Adam and Dropout to be used in the assignment. Generally, the coding parts has 3 parts, which are:\
**(1)** Implement dependency structure by defining the object of ParseModel and initialize the methods like shift, l-arc and r-arc which are taught in the class. After that, we have to access to deal with a minibatch and parse them.\
**(2)** Implement nn.embedding and forward without using the library. Having a good command of torch would help making the program more efficient. for example using .view()this method to flat the matrix. As for the implement of forward, the  important thing is to learn how to use sevaral methods in torch.nn to define avtivate functions and dropout. Also it's important to make it clear about the shape of matrcies in input and hidden layer.\
**(3)** The last part is in run.py, in which most job has been done. In this part, you have to define Adam as our optimizer and cross-entropy as loss function. After that, define the process of training, including how to get to hidden layer and dropout, etc. In this part, you have to learn how to use the functions offered by pytorch and the optimize step as follows.\
for input, target in dataset:\
    {optimizer.zero_grad()\
    output = model(input)\
    loss = loss_fn(output, target)\
    loss.backward()\
    optimizer.step()}

