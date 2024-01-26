import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def NormalizeValues(inputfile, outputfile):
    
    # Loading CSV file into a pandas DataFrame
    df = pd.read_csv(inputfile)
    
    # Extracting numerical columns to normalize
    numerical_columns = df.select_dtypes(include=[float, int]).columns

    # Min-Max scaling to normalize the values between 0 and 1
    scaler = MinMaxScaler()
    df[numerical_columns] = scaler.fit_transform(df[numerical_columns])
    
    # Save the normalized DataFrame back to a CSV file 
    df.to_csv(outputfile, index=False)


NormalizeValues("iris.csv", "normalize_file_iris.csv")
NormalizeValues("letter-recognition.csv", "normalize_file_letter.csv")





