import csv
import sys

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python shopping.py data")

    # Load data from spreadsheet and split into train and test sets
    evidence, labels = load_data(sys.argv[1])
    X_train, X_test, y_train, y_test = train_test_split(
        evidence, labels, test_size=TEST_SIZE
    )

    # Train model and make predictions
    model = train_model(X_train, y_train)
    predictions = model.predict(X_test)
    sensitivity, specificity = evaluate(y_test, predictions)

    # Print results
    print(f"Correct: {(y_test == predictions).sum()}")
    print(f"Incorrect: {(y_test != predictions).sum()}")
    print(f"True Positive Rate: {100 * sensitivity:.2f}%")
    print(f"True Negative Rate: {100 * specificity:.2f}%")


def load_data(filename):
    """
    Load shopping data from a CSV file `filename` and convert into a list of
    evidence lists and a list of labels. Return a tuple (evidence, labels).

    evidence should be a list of lists, where each list contains the
    following values, in order:
        - Administrative, an integer
        - Administrative_Duration, a floating point number
        - Informational, an integer
        - Informational_Duration, a floating point number
        - ProductRelated, an integer
        - ProductRelated_Duration, a floating point number
        - BounceRates, a floating point number
        - ExitRates, a floating point number
        - PageValues, a floating point number
        - SpecialDay, a floating point number
        - Month, an index from 0 (January) to 11 (December)
        - OperatingSystems, an integer
        - Browser, an integer
        - Region, an integer
        - TrafficType, an integer
        - VisitorType, an integer 0 (not returning) or 1 (returning)
        - Weekend, an integer 0 (if false) or 1 (if true)

    labels should be the corresponding list of labels, where each label
    is 1 if Revenue is true, and 0 otherwise.
    """
    col_list = ["Administrative", "Administrative_Duration", "Informational", "Informational_Duration", "ProductRelated", "ProductRelated_Duration", "BounceRates", "ExitRates", "PageValues", "SpecialDay", "Month", "OperatingSystems", "Browser", "Region", "TrafficType", "VisitorType", "Weekend", "Revenue"]

    Administrative = []
    Administrative_Duration = []
    Informational = []
    Informational_Duration = []
    ProductRelated = []
    ProductRelated_Duration = []
    BounceRates = []
    ExitRates = []
    PageValues = []
    SpecialDay = []
    Month = []
    OperatingSystems = []
    Browser = []
    Region = []
    TrafficType = []
    VisitorType = []
    Weekend = []
    Revenue = []

    evidence = [
        Administrative, Administrative_Duration, Informational, Informational_Duration, ProductRelated, ProductRelated_Duration, 
    BounceRates, ExitRates, PageValues, SpecialDay, Month, OperatingSystems, Browser,Region, TrafficType, VisitorType, Weekend
    ]

    file = csv.DictReader(filename)
    for column in file:
        for i in range(len(evidence)):
            if col_list[i] == Month:
                print("going tru month")
                if column[col_list[i]] == "Jan":
                    evidence[i].append(0)
                if column[col_list[i]] == "Feb":
                    evidence[i].append(1)
                if column[col_list[i]] == "Mar":
                    evidence[i].append(2)
                if column[col_list[i]] == "Apr":
                    evidence[i].append(3)
                if column[col_list[i]] == "May":
                    evidence[i].append(4)
                if column[col_list[i]] == "June":
                    evidence[i].append(5)
                if column[col_list[i]] == "Jul":
                    evidence[i].append(6)
                if column[col_list[i]] == "Aug":
                    evidence[i].append(7)
                if column[col_list[i]] == "Sep":
                    evidence[i].append(8)
                if column[col_list[i]] == "Oct":
                    evidence[i].append(9)
                if column[col_list[i]] == "Nov":
                    evidence[i].append(10)
                if column[col_list[i]] == "Dec":
                    evidence[i].append(11)

            elif col_list[i] == VisitorType:
                if column[col_list[i]] == "Returning_Visitor":
                    evidence[i].append(1)
                else:
                    evidence[i].append(0)

            elif col_list[i] == Weekend:
                if column[col_list[i]] == "FALSE":
                    evidence[i].append(0)
                else:
                    evidence[i].append(1)

            else:
                print("going tru", col_list[i])
                evidence[i].append(column[col_list[i]])
            print("end")

    for lbl in file:
        if lbl['Revenue'] == "FALSE":
            Revenue.append(0)
        else:
            Revenue.append(1)
            
    return evidence, Revenue


def train_model(evidence, labels):
    """
    Given a list of evidence lists and a list of labels, return a
    fitted k-nearest neighbor model (k=1) trained on the data.
    """
    kneigh = KNeighborsClassifier(n_neighbors=1)
    trained = kneigh.fit(evidence, labels)
    
    return trained


def evaluate(labels, predictions):
    """
    Given a list of actual labels and a list of predicted labels,
    return a tuple (sensitivity, specificity).

    Assume each label is either a 1 (positive) or 0 (negative).

    `sensitivity` should be a floating-point value from 0 to 1
    representing the "true positive rate": the proportion of
    actual positive labels that were accurately identified.

    `specificity` should be a floating-point value from 0 to 1
    representing the "true negative rate": the proportion of
    actual negative labels that were accurately identified.
    """
    sen_count = 0
    spec_count = 0
    for l, p in labels, predictions:
        if l == 0 and p == 0:
            spec_count = spec_count + 1
        if l == 1 and p == 1:
            sen_count = sen_count + 1

    data_len = len(labels)

    sen_rate = sen_count/data_len
    spec_rate = spec_count/data_len

    return (sen_rate, spec_rate)


if __name__ == "__main__":
    main()
