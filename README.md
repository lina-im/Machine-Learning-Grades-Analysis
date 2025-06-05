# Machine-Learning-Grades-Analysis
 Built two machine learning models, decision forests and neural  networks, based on the Student Performance UC Irvine ML Repository dataset to predict students’ G3 grades. 
# I. Description: 
 The dataset to be used is called, “student-mat,” obtained from Student 
 Performance dataset. The dataset contains 396 rows and 32 columns, and the output 
 target variable is G3. G3 is a numerical variable that represents the student’s final grade 
 in their math course, ranging from 0 to 20. To make this a classification model, if G3 is 
 greater than or equal to the score of 10, the student passes the class denoted by 1. If 
 G3 is less than 10, the student fails the class, denoted by 0. The dataset’s columns 
 describe the student, such as their school, sex, age, address, family size, absences, 
 health status, and more, while the dataset’s rows represent each student taking a math 
 course. The protected variable in this dataset is the students’ sex (male/female), so the 
 test set will be split into these two groups. The dataset was also split into training and 
 testing data, where 75% of the dataset represented training and 25% represented 
 testing.

# II. Brief Description of Methods Used 
 Decision forest is a machine learning model that uses ensemble methods by 
 combining the results of multiple decision trees to improve prediction accuracy and 
 reduce overfitting. Each decision tree is trained on a random subset of data and makes 
 a prediction. For classification tasks, the model takes the majority vote of the tree 
 predictions to determine the final output.  
 
 Neural network is a machine learning model that allows decisions to be made based 
 off patterns. Information is processed through layers while using activation functions 
 and weights. The weights are adjusted using backpropagation to learn complex patterns 
 of large datasets. The model then applies the patterns it learned from the given dataset 
 onto new, unfamiliar dataset to make decisions. 

# III. Results on Entire Test Set with Each of Two ML Methods 
 ![Neural Network](https://github.com/user-attachments/assets/613d4b01-279c-4fd5-a8da-2cf18908039e)

 Figure 1: Decision forest bar graph where the test set was split into female and male 
 because the protected feature was ‘sex.’ 
 
 ![Neural Network 2](https://github.com/user-attachments/assets/57c7b740-1151-46c1-8048-ba61b9692824)
 
 Figure 2: Neural network bar graph where the test set was split into female and male 
 because the protected feature was ‘sex.’ 

 Overall Accuracy of Decision Forest: 0.8889
 
 Overall Accuracy of Neural Network: 0.7374 
 
# VI. Comments and Conclusions 
 Based on the resulting accuracies of the two models, there is slight bias in the decision 
 forest ML classification model and a large bias in the neural network ML classification 
 model. Figure 1 (decision forest) shows that the female accuracy was 0.9184 while the 
 male accuracy was 0.8600. With the difference between the protected variable’s 
 subsets being approximately 0.05, there is a very slight bias present in the decision 
 forest ML classification model. Figure 2 (neural network) shows that the female 
 accuracy was 0.6531 while the male accuracy was 0.8200. With the difference between 
 the protected variable’s subsets being approximately 0.17, there is significant bias 
 present in the decision forest ML classification model in terms of prediction accuracy. 

 This bias can be mitigated by sampling differentially according to the values of the 
 protected variable, sex. From the training sets, 159 counts were female while 137 
 counts were male. To account for the sampling size difference in the subsets, one of the 
 proposed bias mitigation strategy is upsampling. 
