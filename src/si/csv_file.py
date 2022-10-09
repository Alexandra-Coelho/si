# modules
import pandas as pd
import sys
sys.path.insert(0, 'src/si')
from data.dataset import Dataset

def read_csv(filename: str, sep: str = ',',
             features: bool = False, #se a primeira linha do dataset tem o nome das colunas ou se começa logo com dados
             label: bool = False) -> Dataset: #se tem y ou nao
    """
    Reads a csv file into a Dataset object.

    Parameters
    ----------
    filename : str
        Path to the file
    sep : str, optional
        The separator used in the file by default is ','
    features : bool, optional
        Whether the file has a header by default is False
    label : bool, optional
        Whether the file has a label by default is False
    """
    data = pd.read_csv(filename, sep=sep)

    if features and label: #se a 1ºlinha tem o nome das colunas e se tem y 
        features = data.columns[:-1]
        label = data.columns[-1] #y na ultima coluna
        X = data.iloc[1, 0:-1].to_numpy()
        y = data.iloc[1, -1].to_numpy() #passo para numpy porque é um pd dataframe

    elif features and not label: #se nao tem label, mas tem features
        features = data.columns
        X = data.to_numpy() #o dataframe é o X
        y = None

    elif not features and label: #se nao tem features
        X = data.iloc[:, :-1].to_numpy()
        y = data.iloc[:, -1].to_numpy()
        features = None
        label = None

    else: #se nao tem features nem label
        X = data.to_numpy()
        y = None    #modelo nao supervisionado
        features = None
        label = None

    return Dataset(X, y, features=features, label=label)


def write_csv(filename: str, dataset: Dataset, sep: str = ',', features: bool = False, label: bool = False) -> None:
    """
    Writes a Dataset object to a csv file

    Parameters
    ----------
    filename : str
        Path to the file
    dataset : Dataset
        The dataset object
    sep : str, optional
        The separator used in the file, by default ','
    features : bool, optional
        Whether the file has a header, by default False
    label : bool, optional
        Whether the file has a label, by default False
    """
    data = pd.DataFrame(dataset.X) #construir o dataframe 

    if features: #se tem features adiciono às colunas
        data.columns = dataset.features

    if label: #se tem y adiciono ao dataframe, crio a coluna e atribuo o np.ndarray
        data[dataset.label] = dataset.y

    data.to_csv(filename, sep=sep, index=False)  #cria o csv


if __name__ == '__main__':
    df=read_csv(filename="datasets/iris.csv", sep=',')
    print(df.shape())

