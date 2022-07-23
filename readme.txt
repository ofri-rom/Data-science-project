presents:

1.ofri rom : 208891804
2.Avigail shekasta : 209104314
3.Dan Monsonego : 313577595


Python Libraries for models and data pre processing:

1. sklearn (all the models,preprocessing)
2. pandas
3. numpy
4. matplotlib
5. pickle
6. seaborn

Python Libraries for GUI:

1.in this project we use the streamlit GUI to run our project

Dependency:
in this project we have 2 files GUI.py and models_preprocess.py file
the main file is the GUI.py from this file the project starts to run this file imports * from  models_preprocess.py file

files functionallty:

GUI.py -  in this file we have all the GUI component of our project

models_preprocess.py - int his file we have:

function import and install - this function automaticly import and install all the package to run this project.

function get df - this function get the path of the file and create pandas Dataframe with the current file.

function drop rows - this function drop all the rows that contain missing values in the class column.

fill missing value function - function that fill the missing value.

models - id3,knn,K-means,Naive bayes,our id3 model,our Naive bayes model.

functions that creates matrix to visualize the result of the models we ran.

function that save the model result to a binary file we use the pickle to do this.


How to run the program + (instructions):

1.make sure all the packages import and installed

2.in the terminal command prompt type the follow command: streamlit run GUI.py

3.after type the command automaticly the application will open in the web browser

4.type the path of the csv file

5.choose on of to methood to fill the data

6.choose the discretization you want to use(if it needed)

7.choose if you want to use Normalization

8.if you wand to save the clean dataframe click on "save clean data" button and you will see the new file in the project folder

9.choose one of the models (if needed type the hayper parameter for the model)

10. after you choose the model click on "show results of the current model





