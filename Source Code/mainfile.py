#import packages-----------------------------------------------
import seaborn as sns
import pandas as pd
from sklearn.svm import SVC
from sklearn import preprocessing
from sklearn.model_selection import train_test_split 
from sklearn.metrics import confusion_matrix

#1.data slection---------------------------------------------------
dataframe=pd.read_csv("dataset2.csv")
print("-----------------------------------------")
print()
print("Data Selection")
print(dataframe.head())
print()


#2.pre processing--------------------------------------------------
#checking  missing values 
print("---------------------------------------------")
print()
print("Before Handling Missing Values")
print()
print(dataframe.isnull().sum())
print() 

#replace the missing values by 0
median = dataframe['MMSE'].median()
dataframe['MMSE'].fillna(median, inplace=True)
print("-----------------------------------------------")
print("After Handling Missing Values")
print("1.Remove missing values in MMSE------------")
print()
print(dataframe.isnull().sum())
print()

median = dataframe['SES'].median()
dataframe['SES'].fillna(median, inplace=True)
print("-----------------------------------------------")
print()
print("2.Remove missing values in SES------------")
print(dataframe.isnull().sum())
print()


#visulaization---------------------------------------------------
dataframe['Group'] = dataframe['Group'].replace(['Converted'], ['Demented'])
dataframe['Group'] = dataframe['Group'].replace(['Converted'], ['Demented'])
sns.countplot(x='Group', data=dataframe)

#label encoding
#Encode columns into numeric
print("------------------------------------------------------")
print()
print("Before Label Encoding")
print()
print(dataframe['Group'].head())
label_encoder = preprocessing.LabelEncoder() 
print("------------------------------------------------------")
print()
print("After Label Encoding")
print()
dataframe['Group']= label_encoder.fit_transform(dataframe['Group'])
print(dataframe['Group'].head()) 
dataframe['M/F']= label_encoder.fit_transform(dataframe['M/F']) 
dataframe['Hand'] = label_encoder.fit_transform(dataframe['Hand'])


#3.data splitting--------------------------------------------------
feature_col_names = ["M/F", "Age", "EDUC", "SES", "MMSE", "eTIV", "nWBV", "ASF"]
predicted_class_names = ['Group']


X = dataframe[feature_col_names].values
y = dataframe[predicted_class_names].values

#spliting the x and y into test and train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=2)


#4.classification algorithms------------------------------------------
#svm
svm = SVC(kernel="linear", C=0.1,random_state=0)
svm.fit(X_train, y_train.ravel())
pred = svm.predict(X_test)

#confusion matrix
print("-----------------------------------------------------")
print("Performance Metrics")
cm1=confusion_matrix(y_test,pred)
print()
print("1.Confusion Matrix",cm1)
print()

#find the performance metrics 
TP = cm1[0][0]
FP = cm1[0][1]
FN = cm1[1][0]
TN = cm1[1][1]

#Total TP,TN,FP,FN
Total=TP+FP+FN+TN

#Accuracy Calculation
accuracy1=((TP+TN)/Total)*100
print("2.Accuracy",accuracy1,'%')
print()

#Precision Calculation
precision=TP/(TP+FP)*100
print("3.Precision",precision,'%')
print()

#Sensitivity Calculation
Sensitivity=TP/(TP+FN)*100
print("4.Sensitivity",Sensitivity,'%')
print()

#specificity Calculation
specificity = (TN / (TN+FP))*100
print("5.specificity",specificity,'%')
print()



#predict the disease 

if y[0]== 0:
    print("------------------------------")
    print('\n Non Dementia ')
    print("------------------------------")
else:
    print("------------------------------")
    print('\n Demtia ')
    print("------------------------------")

















































