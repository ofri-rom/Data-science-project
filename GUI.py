# presents: ofri rom:208891804,Avigail shekasta:209104314,Dan monsonego:313577595

# GUI files with streamlit
# ****************** import libs ***************** #

import streamlit as st
import seaborn as sns

from models_preprocess import *


# function to create our GUI component
def create_web_page():
    colc, coll = st.columns(2)
    with colc:
        st.title('Final project')
    st.title(' ')
    # input path
    st.subheader("Please enter the path of the csv file:")
    csv_path = st.text_input('input path', )
    classification = st.text_input('Enter the classification column')
    # end of input path

    # fill missing values
    right, left = st.columns(2)
    with right:
        if st.button('fill missing values by class column'):
            get_df(csv_path)
            drop_rows(classification)
            main_fill_data(2, classification)
            Conversion_to_number()

    with left:
        if st.button('fill missing values in relative to all data'):
            get_df(csv_path)
            drop_rows(classification)
            main_fill_data(1, classification)
            Conversion_to_number()

    # discretiziation choose
    st.subheader("Please choose the discretization you want to use:")
    Bins = st.text_input('Enter the bins for that action')
    col_nameD = st.text_input('Enter the name of the column to perform discretiziation ')

    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button('Equal with'):
            Equal_width(col_nameD, int(Bins))
    with col2:
        if st.button('Equal frequency'):
            Equal_frequency_discretization(col_nameD, int(Bins))
    with col3:
        if st.button('Entropy based'):
            entrophy_based_binning(col_nameD, int(Bins))
    # end of discretization choose

    # Normalization choose
    st.subheader("Please choose if you want to use Normalization:")
    col_nameN = st.text_input('Enter the name of the column to perform Normalization')
    if st.button('Yes'):
        Normalization(col_nameN)
    # end of Normalization choose

    # *************************************save the clean data********************************************
    if st.button('save clean data'):
        save()
    # *************************************end of save the clean data********************************************

    # model choose to perform
    st.subheader("Please choose the models you want to run on the data:")
    parameter = st.text_input('Please choose the parameter tuning for this model/notice:by default the parameter is 5')
    r1, l1 = st.columns(2)
    with r1:
        if st.button('Id3'):
            id3(classification)
            pickl_model_save()
    with l1:
        if st.button('Id3 by us'):
            le = preprocessing.LabelEncoder()
            for i in d['df'].columns:
                d['df'][i] = le.fit_transform(d['df'][i])
            X = d['df']
            y = d['df']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            print(y_test)
            tree = id3_by_us(X_train, X_train, d['df'].columns[1:], classification)
            print(tree)
            positive, negative = test(X_test, tree)
            print('Postive test-', positive)
            print('Negative test-', negative)
            pickl_model_save()
    r, l = st.columns(2)
    with r:
        if st.button('Naive Bayes'):
            Naive_bayes(classification)
            pickl_model_save()
    with l:
        if st.button('Our Naive Bayes'):
            Naive_bayes_by_us(classification)
            pickl_model_save()

    if st.button('Knn'):
        Knn(classification, int(parameter))
        pickl_model_save()
    if st.button('K-means'):
        kmeans(classification, int(parameter))
        pickl_model_save()
    st.subheader("Model results and confusion matrix:")
    if st.button('Show results of the current model'):
        plt.figure(figsize=(7, 6))
        sns.heatmap(matrix_performace(), annot=True, fmt='d')
        st.pyplot(fig=plt, clear_figure=None)
        st.subheader('Results acc = ' + str(acc()))
        st.subheader('Results recall = ' + str(recall()))
        st.subheader('Results f measure = ' + str(fmeasure()))
        st.subheader('Results precision = ' + str(precision()))
    if st.button('train matrix'):
        plt.figure(figsize=(7, 6))
        sns.heatmap(matrix_train(), annot=True, fmt='d')
        st.pyplot(fig=plt, clear_figure=None)
    if st.button('test matrix'):
        plt.figure(figsize=(7, 6))
        sns.heatmap(matrix_test(), annot=True, fmt='d')
        st.pyplot(fig=plt, clear_figure=None)
    if st.button("Finish"):
        matrix_performace()
        matrix_test()
        matrix_test()
        majority_test(classification)
        pickl_matrix_etc()


def main():
    create_web_page()


main()
