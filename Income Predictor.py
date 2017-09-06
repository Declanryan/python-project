import httplib2
import string
import time
from tkinter import *
import matplotlib.pyplot as plt
import numpy as np

# by Declan Ryan

'''Using a dataset ( the "Adult Data Set") from the UCI Machine-Learning Repository we
can predict based on a number of factors whether someone's income will be greater than $50,000.

The technique:
The approach is to create a 'classifier' - a program that takes a new example record and, based
on previous examples, determines which 'class' it belongs to. In this problem we consider attributes
of records and separate these into two broad classes, <50K and >=50K. We begin with a training data
set - examples with known solutions. The classifier looks for patterns that indicate classification.
These patterns can be applied against new data to predict outcomes. If we already know the outcomes
of the test data, we can test the reliability of our model. If it proves reliable we could then use
it to classify data with unknown outcomes. We must train the classifier to establish an internal model
of the patterns that distinguish our two classes. Once trained we can apply this against the test data
- which has known outcomes. We take our data and split it into two groups - training and test - with most
of the data in the training set.

Building the classifier
Look at the attributes and, for each of the two outcomes, make an average value for each one, Then average
these two results for each attribute to compute a midpoint or 'class separation value'. For each record,
test whether each attribute is above or below its midpoint value and flag it accordingly. For each record
the overall result is the greater count of the individual results (<50K, >=50K). You should track the
accuracy of your model, i.e how many correct classifications you made as a percentage of the total number
of records.'''


def get_url_text(url):
    ''' This function is used to get the data from the web decode from byte and split on the end of line character'''
    try:
        h = httplib2.Http(".cache")
        headers, body = h.request(url)
        body = body.decode().split("\r")
        return headers, body

    except httplib2.HttpLib2Error as e:
        print(e)
        return False


def file_creation(file_entered):
    '''Splits the file into a training file and a testing file and checks for bad data
    :param file_entered: file to be split
    :return: the two seperated files and bad records
    '''
    new_file_list = []
    bad_records = 0
    for line in file_entered:
        line.strip('\n')
        if '?' in line:
            bad_records += 1
            continue
        elif len(line) < 14:
            bad_records += 1
            continue
        new_file_list.append(line)
    count = 0
    training_file_list = []
    testing_file_list = []
    for line_str in new_file_list:
        line_str.strip(string.whitespace).strip(string.punctuation)
        if count <= int((len(file_entered))/100*75):
            training_file_list.append(line_str)
        elif int((len(file_entered))/100*75) < count <= len(file_entered):
            testing_file_list.append(line_str)
        else:
            break
        count += 1
    return training_file_list, testing_file_list, bad_records


def make_data_set(list_entered):
    '''  iterates through the list and calculates the number of times certain strings appear by adding it to a
     dictionary and adding one each time it appears again. the list is looped through again and each row is split at the
     ',' . A tupple is formed for each record with numerical values for the strings, the tupple is then added to a new
      list and the list is returned
    :param list_entered: file to be changed
    :return: list of tuples
    '''

    working_class_dict = {}
    marital_status_dict = {}
    occupation_dict = {}
    relationship_dict = {}
    race_dict = {}
    sex_dict = {}
    data_list = []

    for row in list_entered:
        row = row.split(",")
        if row[1] in working_class_dict:
            working_class_dict[row[1]] += 1
        else:
            working_class_dict[row[1]] = 1
        if row[5] in marital_status_dict:
            marital_status_dict[row[5]] += 1
        else:
            marital_status_dict[row[5]] = 1
        if row[6] in occupation_dict:
            occupation_dict[row[6]] += 1
        else:
            occupation_dict[row[6]] = 1
        if row[7] in relationship_dict:
            relationship_dict[row[7]] += 1
        else:
            relationship_dict[row[7]] = 1
        if row[8] in race_dict:
            race_dict[row[8]] += 1
        else:
            race_dict[row[8]] = 1
        if row[9] in sex_dict:
            sex_dict[row[9]] += 1
        else:
            sex_dict[row[9]] = 1

    for row in list_entered:
        row = row.split(",")
        length = len(list_entered)
        data_tuple = int(row[0]), working_class_dict[row[1]] / length, int(row[4]), \
            marital_status_dict[row[5]]/length, (occupation_dict[row[6]]/length), (relationship_dict[row[7]]/length), \
            (race_dict[row[8]] / length), (sex_dict[row[9]] / length), int(row[10]), int(row[11]), int(row[12]), row[14]

        data_list.append(data_tuple)
    return data_list


def sum_of_lists(list_1, list_2):
    '''Itreates through each list and adds the elements of the two lists entered together
    :param list_1: first list entered
    :param list_2: second list entered
    :return: Total of the two lists
    '''
    total_list = []
    for i in range(11):
        total_list.append(list_1[i] + list_2[i])
    return total_list


def get_average(list_entered, total_int):
    '''Averages the elements of the list by dividing each
       element by the total

    :param list_entered: file to be iterated through
    :param total_int: total for division
    :return: list of average values
    '''
    average_list = []
    for value_int in list_entered:

        average_list.append(value_int/total_int)
    return average_list


def train_classifier(list_entered):
    '''Splits the list into greater and less than lists,
       then we use the average function to obtain average values of these lists
       and the sum of function to add the lists together and get the average.

    :param list_entered:list to be used
    :return: the result of the averages and summation
    '''
    greater_than_50_list = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, " "]
    greater_count = 0
    less_than_50_list = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, " "]
    less_count = 0
    for person in list_entered:

        if person[-1] == ' <=50K':
            less_than_50_list = sum_of_lists(less_than_50_list, person)
            less_count += 1
        else:
            greater_than_50_list = sum_of_lists(greater_than_50_list, person)
            greater_count += 1

    less_than_average_list = get_average(less_than_50_list, less_count)
    greater_than_average_list = get_average(greater_than_50_list, greater_count)

    classifier_list = get_average(sum_of_lists(less_than_average_list, greater_than_average_list), 2)
    return classifier_list


def classify_test_set_list(test_set_list, classifier_list):
    ''' checks each index in the tuple against its corresponding averaged value from the classifier list and keeps
     count of the greater and less than value. We then create a new tuple from the count of the greater and less than
     values and the result string, we then add it to a new list.

    :param test_set_list:the list of tupples to be checked
    :param classifier_list:the list against which the tuple is compared
    :return:list of tuples with 3 elements
    '''
    result_list = []
    for person_tuple in test_set_list:
        less_than_count = 0
        greater_than_count = 0
        result_str = person_tuple[-1]
        for index in range(11):
            if person_tuple[index] > classifier_list[index]:
                greater_than_count += 1
            else:
                less_than_count += 1
        result_tuple = (less_than_count, greater_than_count, result_str)
        result_list.append(result_tuple)
    return [result_list]


def report_results(result_set_list):
    ''' sorts the results into the correct category and displays the total count of records used,
     the no of inaccurate records and the percentage accuracy rate '''
    total_count = 0
    less_than_total = 0
    greater_than_total = 0
    equal_results = 0
    inaccurate_count = 0
    unclassified = 0
    for first_list in result_set_list:
        for result_tuple in first_list:
            less_than_count, greater_than_count, result_str = result_tuple[:3]
            total_count += 1
            if (less_than_count < greater_than_count) and (result_str == ' <=50K'):
                inaccurate_count += 1
            elif(greater_than_count < less_than_count) and (result_str == ' >50K'):
                inaccurate_count += 1
            elif(less_than_count > greater_than_count) and (result_str == ' <=50K'):
                less_than_total += 1
            elif(less_than_count == greater_than_count) and (result_str == ' <=50K'):
                equal_results += 1
            elif(greater_than_count > less_than_count) and (result_str == ' >50K'):
                greater_than_total += 1
            elif(greater_than_count == less_than_count) and (result_str == ' >50K'):
                equal_results += 1
            else:
                unclassified += 1

    one_percent = total_count / 100
    percentage = (total_count - inaccurate_count) / one_percent
    tot_rec = ("Out of {} records, there were {} inaccuracy's \n {} records had <50K, {} had >50K, {} Equal results\n"
           " Unclassified records is {}\n That gives a percentage accuracy of %{:.2f} ".format(total_count,
                inaccurate_count, less_than_total, greater_than_total, equal_results, unclassified, percentage))

    # plots a bar chart representation of the results
    N = 6  # no of columns
    a = total_count
    b = inaccurate_count
    c = less_than_total
    d = greater_than_total
    e = equal_results
    f = unclassified
    moves = (total_count)
    scores = (a, b, c, d, e,f)
    ind = np.arange(N)  # the x locations for the groups
    width = 0.2  # the width of the bars: can also be len(x) sequence
    plt.bar(ind, scores, width)
    plt.ylabel('No of records')  # y axis label
    plt.title('Results of testing file')  # title
    plt.xticks(ind, ('Total Count', "Inaccuracy's", '<50k', '>50k', 'equal', 'unclassified'))  # x axis label
    plt.yticks(np.arange(0, moves, 500))  # size of y axis and increments

    return tot_rec

def start(root):
    try:  # The get url function gives us http header information and the content as byte.
        # the file is then decoded into a list of str that has been split using the last char(\r)
        header, contents_list = get_url_text("http://mf2.dit.ie/machine-learning-income.data")

        # time is used two keep a check on the running time of the program
        start_time = time.time()

        # section 1: of the program is used to check the no of lines in the program, it then creates two files from the
        # original one for training(75%) and one for testing(25%). The program outputs how many lines are in each file
        # and the total number of bad records

        total_files = Label(root, text="Total records in the main file is {} \n Creating the two files".format(
            len(contents_list)))
        training_file_list, testing_file_list, bad_records = file_creation(contents_list)
        success = Label(root, text="Files created successfully")
        records = Label(root, text="Total bad records from the file is {}".format(bad_records))
        training_records = Label(root, text='The Training file has {} records'.format(len(training_file_list)))
        testing_records = Label(root, text='The Testing file has {} records'.format(len(testing_file_list)))
        stars3 = Label(root, text="*" * 60)

        # section 2:- of the program uses the make data set function to create a training list of tuples
        read_train_data = Label(root, text="Reading in training data...")
        training_list = make_data_set(training_file_list)
        fin_train_read = Label(root, text="Done reading training data.")
        stars4 = Label(root, text="*" * 60)

        # Section 3:- of the program is used to train the classifier , which involves Splitting the list into greater
        # and less than lists. Then we use the average function to obtain average values and the sum of function to
        # add the lists together.
        train_class = Label(root, text="Training classifier...")
        classifier_list = train_classifier(training_list)
        fin_train_class = Label(root, text="Done training classifier.")
        stars5 = Label(root, text="*" * 60)

        # section 4:- of the program uses the make data set function to create a testing list of tuples
        read_test_data = Label(root, text="Reading in test data...")
        test_set_list = make_data_set(testing_file_list)
        fin_test_data = Label(root, text="Done reading testing data.")
        stars6 = Label(root, text="*" * 60)

        # section 5 of the program uses the classify test set function to check each index in the tuple against its
        # corresponding averaged value from the classifier list. It then creates a list of tuples with 3 elements
        # the less than count, greater than count and the result_str

        start_classifying = Label(root, text="Classifying records...")
        result_str = classify_test_set_list(test_set_list, classifier_list)
        fin_classifying = Label(root, text="Done classifying.")
        stars7 = Label(root, text="*" * 60)

        # section 6:- is used to sort the results in to there correct category and output the results
        print_result = Label(root, text="Printing results")
        tot_rec_ret = Label(root, text=report_results(result_str))
        total_time = Label(root, text="The program took {:.2f} sec to run".format(time.time() - start_time))
        end = Label(root, text="Program finished.")
        stars8 = Label(root, text="*" * 60)
        stars9 = Label(root, text="*" * 60)

        # packs all the labels into the frame
        total_files.pack(), success.pack(), records.pack()
        training_records.pack(), testing_records.pack(), stars3.pack(), read_train_data.pack(), fin_train_read.pack()
        stars4.pack(), train_class.pack(), fin_train_class.pack(), stars5.pack(), read_test_data.pack()
        fin_test_data.pack(), stars6.pack(), start_classifying.pack(), fin_classifying.pack(), stars7.pack()
        print_result.pack(), tot_rec_ret.pack(), total_time.pack()
        end.pack(), stars8.pack(), stars9.pack()

    # the except is used to catch any io error or value error within the main function and output the error
    except IOError as e:
            print(e)
            quit()

    except ValueError as e:
        print(e)
        quit()

def main():
    ''' The main function calls the functions in the order required.'''

    root = Tk()
    root.title("Income Predictor")
    space = Label(root, text=" ")
    space.pack()
    url = StringVar()
    url.set("http://mf2.dit.ie/machine-learning-income.data")
    enter_label = Label(root, padx=5, pady=5, text="URL for data")
    enter_label.pack()
    url_box = Entry(root, textvariable=url)
    url_box.focus_set()
    start_btn = Button(root, text="START", command=start, padx=5, pady=5, bg="black", fg="white")
    start_btn.pack()
    stars1 = Label(root, text="*" * 60)
    stars1.pack()
    stars2 = Label(root, text="*" * 60)
    stars2.pack()
    start(root)

    # shows the frame and chart
    plt.show()
    root.mainloop()

# Run if stand-alone
if __name__ == '__main__':
    main()
