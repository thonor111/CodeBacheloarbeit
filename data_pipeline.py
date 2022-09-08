import os
import pandas as pd



def preprocess_LKOS_data(filename="Fallzahlen pro Tag.xlsx", outputname="preprocessedLKOS.csv", ID_to_name=False):
    cd = os.getcwd()  # has to be adapted for final data structure
    file = cd + "/" + filename
    data = pd.read_excel(file)  # , encoding='latin-1')
    data

    # change names to ids and drop natural names (but save them for later)
    data["ID"] = list(range(len(data.index)))
    data.set_index("ID", drop=True, inplace=True)
    id_to_name = data[["Bestätigte (Neu-)Fälle pro Tag"]]
    id_to_name.rename(columns={"Bestätigte (Neu-)Fälle pro Tag": "NL Name"}, inplace=True)
    data.drop(["Bestätigte (Neu-)Fälle pro Tag", "Summe"], axis=1, inplace=True)

    data.columns = pd.to_datetime(data.columns, dayfirst=True)
    df = data.transpose()

    df.to_csv(cd + "/" + outputname)
    if ID_to_name:
        id_to_name.to_csv(cd + "/ID_to_name.csv")
    print("Successfully saved newest data in appropriate form.")
    return data


def split_data(
        data,
        train_start,
        test_start,
        post_test
):
    """
        split_data(data,data_start,train_start,test_start)

    Utility function that splits the dataset into training and testing data as well as the corresponding target values.

    Returns:
    ========
        data_train:     training data (from beginning of records to end of training phase)
        target_train:   target values for training data
        data_test:      testing data (from beginning of records to end of testing phase = end of records)
        target_test:    target values for testing data
    """
    # print("\ntrain_start", train_start, "\ntest_start", test_start, "\npost test",post_test)
    target_train = data.loc[(train_start <= data.index)
                            & (data.index < test_start)]
    target_test = data.loc[(test_start <= data.index)
                           & (data.index < post_test)]

    data_train = data.loc[data.index < test_start]
    data_test = data

    return data_train, target_train, data_test, target_test


def load_data_n_weeks(
        csv_path,
        start_day,
        seperator=",",
        pad=None
):
    ''' loads the data starting at a given timepoint
    Arguments:
        start (int): Days after '2020-03-05' to start the data (adapted for new date in LKOS data) NOT ENTIRELY SURE WHY WE WOULD START LATER TBH
        csv_path (str): Path to the file, inclusing the file name
        pad (int): How many days are going to be added (nan filled) at the end
    Returns:
        data (pd.df): Daframe with date as index, columns with countie IDs and values in cells.
    '''

    data = pd.read_csv(csv_path, sep=seperator, encoding='iso-8859-1', index_col=0)

    data.index = [pd.Timestamp(date) for date in data.index]

    data = data.loc[start_day <= data.index]  # changed start_day to start

    if pad is not None:
        last_date = data.index[-1]

        extended_index = pd.date_range(last_date + pd.Timedelta(days=1),
                                       last_date + pd.Timedelta(days=pad))
        for x in extended_index:
            data = data.append(pd.Series(name=x))

    data.index = [pd.Timestamp(date) for date in data.index]

    return data