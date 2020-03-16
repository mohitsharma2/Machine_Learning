"""
ml-ai-01.jpg

Gang of Four (Yann Lecun, Geoffrey Hilton, Yoshua Bengio, Andrew Ng )
    CoFounder of Coursera and Baidu (Andrew Ng )
    How to boost GDP of India with Prime minister  ?
    Father of Machine Learning ( Geoffrey Hilton )
    Head of Google AI team since 2001
    Research Student of Hilton (Yann Lecun )
    Heading Facebook AI Team ( Yann Lecun )
    Father of Deep Learning
    Yoshua Bengio is purely Academician, writes Books and Papers
    

ml-ai-02.png
Define Artificial Intellegence ( AI )
    Any machine which can behave like humans (for a specific task) is a AI Machine
    Narrow AI
    Generalized AI
    There are two ways to achieve it ( Machine Learning and Deep Learning )
    Deep Learning is a subset of Machine Learning using Neueral Networks
    Neural Network is a way of computation 
    

Define Data Analytics
    Inference from past Data
    Give Example of Placement data from college
    
    
Types of Analytics
    Descriptive  - What Happened
    Predictive   - What will happen
    Prescriptive - What to do 



Introduce Machine Learning
    Based on past data if we can predict whether Sandeep would be placed ?
    
      
Define BigData
    If the dataset is big, which cannot be processed on your local system
    Distributed computation or Parallel Computing on a cluster is the solution 
    to solve the Big Data Problem
    Frameworks - Apache Hadoop and Spark and now Datbricks
    Map and Reduce is the base concept developed by an Indian(Sanjay Ghemawat)
    in Google

Define Streaming Analytics
    When the data is changing at runtime, like Facebook Like and Comments
    Kafka, elastic search Solr and Lucene 

Define NLP
    Natural Language Processing 
    Ever evolving language 
    
    
Define Data Science 
    Anything around data is known as data science

Define RPA
    Robotic Process Automation    
    UIPath
    

Define Data Scientist
    Dr. D G Patil, First Data Scientist in Obama Administration.
    He coined the word Data Scientist 
    Sexiest Job in the world is of a Data Scientist 
    Understanding of the domain also with the tools and technologies
    Health Sector, Insurance, Retail, Logistics, Banking and Finance
    As a fresher we dont have domain knowledge so we are not Data Scientist
    


Define Machine Learning
    In mathematics
    
    y = f(x) 
    y is output variable / dependent variable
    x is input variable / independent variable 
    f is known as a mapping function / Model 
    ML = Process of training the Model for prediction on never-before-seen dataset
    

    In a traditional system you have x and the mapping function f()
    and you generate the values of y

    In a ML system, you have x and y and based on that we find the mapping function f()
    Once you know the mapping function, 
    we can then give new x values which will predict y
    
    
    In Machine Learning 
        x = Feature
        y = Label / Response / Target
        
    f(newx) ==> y - prediction
    x = [1,2,3,4,5] {6}
    y = [1,2,3,4,5] ??


    To Summarize
    
    Label    = variable we're predecting, typically 
              represented by variable y

    Feature  = Input variables describing the data, 
              represented by variables {x1,x2,x3....xn}

    Model    = Piece of Software or Mapping function 
              ( maps examples to predicted labels )

    ML       = Process of training the Model for prediction 
              on never-before-seen dataset

    Training = Feeding the features and their corresponding labels into an algorithm


    Prediction can be of two types 
    1. Continuous values ( Regression Model )
    2. Discrete Values   ( Classification Model )


Define Dataset
    Collection of and/or features and label is called dataset

    If both feature(x) and label(y) is present in the data, 
    it is known as Labelled Dataset
    
    show student_scores.csv
    
    If only feature (x) is present and label (y) is missing in a data, 
    its a non labelled dataset
    
    
Types of Machine Learning    
    Supervised ML works on Labelled dataset for predictions
    
    Unsupervised ML works on unlabelled data.
    Predictions are not possible, only groups can be created ( clustering )
    

    Supervised ML ( Predictions ) 
    Prediction can be of two types 
    1. Continuous values ( Regression Model )
    2. Discrete Values   ( Classification Model )

    Regression - Linear, Multiple , Polynomial
    Classification - kNN, 
    
    Unsupervised ML ( Clustering ) has two branches Clustering and Association 
    Clustering - kmeans
    Association - Apriori
    
    Will netaji WIN or LOOSE in the election is an example of Classification problem 
    With how many votes will he WIN or LOOSE is an example of Regression problem
    
    
Examples of ML    
    Give ( Classification )Example of Spam Detection of Email to explain the below concept
    In the spam detector example, the features could include the following:
    (feature)          words in the email text
    (feature)          sender's address
    (feature)          time of day the email was sent
    (feature)          email contains the phrase "one weird trick."
    (label )           Whether the email is SPAM or HAM

    Another ( Regression )Example is the Housing Prices in Jaipur
    housingMedianAge            (feature)
    totalRooms                  (feature)
    totalBedrooms               (feature)
    medianHouseValue            (label)

    Example of amateur botanist ( to define features and label)
    Leaf Width 	Leaf Length 	Species
    2.7 	    4.9 	        small-leaf
    3.2 	    5.5 	        big-leaf
    2.9 	    5.1 	        small-leaf
    3.4 	    6.8 	        big-leaf

    Leaf width and leaf length are the features (which is why they are both labeled X), 
    while the species is the label.

    Features are measurements or descriptions; the label is essentially the "answer."

    For example, the goal of the data set is to help other botanists answer the question, 
    "Which species is this plant?"

    How ML powers Google Photos: 
        Find a specific photo by keyword search without manual tagging
        ML powers the search behind Google Photos to classify people, places, and things.

    Smart Reply Feature of Gmail


    Real World Example of Supervised Learning
    Study from Stanford University to detect skin cancer in images
    training set contained images of skin labeled by dermatologists as having one of several diseases. 
    The ML system found signals that indicate each disease from its training set
    and used those signals to make predictions on new, unlabeled images.


    Scikitlearn (sklearn) is the library for ML Algorithms


What we can do with ML/DL/AI ?
    Detection
    Classification
    Segmentation
    Prediction
    Recommendation


ML hands On
    Hands On 1:
        Suppose you want to develop a supervised machine learning model to predict 
        whether a given email is "spam" or "not spam." 
        Which of the following statements are true? 

    1. We'll use unlabeled examples to train the model. ( FALSE)
    2. Emails not marked as "spam" or "not spam" are unlabeled examples ( TRUE)
    3. Words in the subject header will make good labels. ( FALSE )
    4. The labels applied to some examples might be unreliable. ( TRUE )


    Hands On  2:
        Suppose an online shoe store wants to create a supervised ML model that will 
        provide personalized shoe recommendations to users. 
        That is, the model will recommend certain pairs of shoes to Marty and different 
        pairs of shoes to Janet. 
        Which of the following statements are true? 
    
    1. "Shoe beauty" is a useful feature. ( FALSE )
    2. "The user clicked on the shoe's description" is a useful label. ( TRUE )
    3. "Shoe size" is a useful feature. ( TRUE )
    4. "Shoes that a user adores" is a useful label. ( FALSE )




ML Problems
    
    Type of ML Problem	Description	                  Example
Classification 	    Pick one of N labels 	      Cat, dog, horse, or bear

Regression 	        Predict numerical values 	  Click-through rate

Clustering 	        Group similar examples 	      Most relevant documents (unsupervised)

Association         Infer likely association      If you buy hamburger buns,
rule learning 	    patterns in data 	          you're likely to buy hamburgers (unsupervised)

Structured output 	Create complex output 	      Natural language parse trees, image recognition bounding boxes

Ranking 	        Identify position on a  	      Search result ranking
                    scale or status


    
Define Deep Learning
    When the dataset is big we would the neural networks to solve it
    Supervised Deep Leaarning ( ANN, RNN, CNN )
    UnSupervised Deep Leaarning ( SOM, AutoEncoders )  
    SOM (Self Organising Maps) is used for Clustering
    AutoEncoders is used for developing Recommendation Systems
    Keras based on Tensorflow is the library for DL Algorithms
    Another library is PyTorch from Facebook/Uber


Define Reinforcement Learning
    In Reinforcement Learning ( RL )
    In RL you don't collect examples with labels.
    Imagine you want to teach a machine to play a very basic video game and never lose. 
    You set up the model (often called an agent in RL) with the game, and you tell 
    the model not to get a "game over" screen. 
    During training, the agent receives a reward when it performs this task, 
    which is called a reward function. With reinforcement learning, 
    the agent can learn very quickly how to outperform humans. 

    However, designing a good reward function is difficult, 
    and RL models are less stable 
    and predictable than supervised approaches.


List of Website
    https://machinelearningmastery.com
    https://www.pyimagesearch.com
    https://www.analyticsvidhya.com
    https://towardsdatascience.com
    Stat Quest (https://www.youtube.com/channel/UCtYLUTtgS3k1Fg4y5tAhLbw)


Mind Map

                                Machine Learning
                                /           \
                               /             \
                              /               \
                      Supervised             Unsupervised
                       /  \                   /       \
                      /    \                 /         \
                     /      \               /           \
                    /        \             /             \ 
           Regression     Classification  Clustering    Association
           |-SLR          |-Logistic       |-kmeans      |-Apriori
           |-MLR          |-kNN            |-Dbscan
           |-PLR          |-SVM            |-Hierarical
           |-DS           |-DS
           |-RF           |-RF
           |-SVR          |-Naive Bayes


SLR = Simple Linear Regression
MLR = Multiple Linear Regression
PLR = Polynomial Linear Regression
DS  = Decission Tree 
RF  = Random Forest 
SVR = Support Vector Regression
SVM = Support Vector Machines 


                               Deep Learning
                                /           \
                               /             \
                              /               \
                      Supervised             Unsupervised
                       /  \                   /       \
                      /    \                 /         \
                     /      \               /           \
                    /        \             /             \ 
           Regression     Classification  Clustering    Association
           |-ANN          |-ANN            |-SOM          
           |-CNN-ANN                       |-Boltzman Machine
           |-RNN-ANN                       


Pre trained models - OpenCV, Yolo, VGG, Resnet,
Dimension Reduction - PCA, Auto Encoders

ML - anaconda, Spyder, Jupyter
DL - Google Colab (GPU, TPU)

Python, Pandas, Numpy, matplotlib, scikitlearn
Keras
Tensorflow
PyTorch - FB, Uber
MxNET    
"""


