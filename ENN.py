import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import EditedNearestNeighbours
from sklearn.preprocessing import LabelEncoder

def ENN(inputfile, outputfile):
    # Loading  CSV file into a pandas DataFrame
    df = pd.read_csv(inputfile)

    # Assuming the last column is the target variable (class label) -> in our case is 'class' inside the CSV
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    # Converting class labels to numerical format using LabelEncoder
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Applying ENN algorithm with library EditedNearestNeighbours
    enn = EditedNearestNeighbours()
    X_resampled, y_resampled = enn.fit_resample(X_train, y_train)

    # Converting back to the original class labels 
    y_resampled = label_encoder.inverse_transform(y_resampled)

    # Converting the resampled data to a DataFrame
    df_resampled = pd.DataFrame(data=X_resampled, columns=df.columns[:-1])
    df_resampled['class'] = y_resampled  # Assuming 'class' is the name of the target variable column

    # Saving the cleaned data to a new CSV file
    df_resampled.to_csv(outputfile, index=False)


ENN("normalize_file_iris.csv", "ENN_iris.csv")
ENN("normalize_file_letter.csv", "ENN_letter.csv")