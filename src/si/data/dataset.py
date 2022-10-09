import numpy as np
import pandas as pd

class Dataset:
    '''Este objeto é responsavel por guardar e representar um dataset de machine learning que contem features (atributos), labels (classes), vetor y '''
    
    def __init__(self, X, y=None, features=None, label=None):
        self.X = X  #numpy ndarray
        self.y = y  #array de uma dimensao
        self.features = features  #lista de strings
        self.label = label  #string

    def shape(self) -> tuple:
        """
        Retorna um tuplo com as dimensões do dataset (número de exemplos e de features)
        """
        return self.X.shape

    def has_label(self) -> bool:
        """
        Retorna um valor booleano para verificar a presença da variável dependente
        """
        if self.y is not None:  #se o y existe entao é True, é supervisionado
            return True
        return False

    def get_classes(self) -> np.ndarray:
        """
        Retorna um np.ndarray (os valores únicos possiveis) com as classes do dataset (y)
        """
        #se nao tem label nao posso ir buscar as classes (meter um raise valueerror pex), porque o dataset é nao supervisionado
        if self.y is None: #num dataset nao supervisionado dá None, nao tem label
            return
        return np.unique(self.y) #if self.has.label() retorno isto

    def get_mean(self) -> np.ndarray:
        """
        Retorna um np.ndarray com a média para cada feature do dataset
        """
        return np.mean(self.X, axis = 0) #axis=0 para colunas, axis=1 para linhas

    def get_variance(self) -> np.ndarray:
        """
        Retorna um np.ndarray com a variância para cada feature do dataset
        """
        return np.var(self.X, axis = 0)

    def get_median(self) -> np.ndarray:
        """
        Retorna um np.ndarray com a mediana para cada feature do dataset
        """
        return np.median(self.X, axis = 0)
    
    def get_min(self) -> np.ndarray:
        """
        Retorna um np.ndarray com o valor mínimo para cada feature do dataset
        """
        return np.min(self.X, axis = 0)
    
    def get_max(self) -> np.ndarray:
        """
        Retorna um np.ndarray com o valor máximo para cada feature do dataset
        """
        return np.max(self.X, axis = 0)

    def summary(self):
        return pd.DataFrame(
            {'mean': self.get_mean(),
            'median': self.get_median(),
            'variance': self.get_variance(),
            'min': self.get_min(),
            'max': self.get_max()}
        )

    def dropna(self):
        self.X = self.X[~np.isnan(self.X).any(axis=1)] #~ faz o oposto, retorna todas as linhas que nao têm NaN preservando o shape
        if self.has_label() is True and self.X.shape[0] == len(self.y): #se tem y, tenho de fazer a mesma verificaçao
            self.y = self.y[~np.isnan(self.X).any(axis=1)]  #nºexemplos=nºelementos de y

    def fillna(self, value):
        self.X[np.isnan(self.X)] = value #no indice onde é True (é NaN) substitui isso por um valor
        return self.X   # ou np.nan_to_num(self.X, copy=False, nan = value)
    

if __name__ == '__main__':
    x = np.array([[1,2,3], [1,2,3], [1,2,6]])  #matriz
    y = np.array([1,2,2]) #vetor
    features = ['A', 'B', 'C']
    label = 'y'  #nome do vetor
    dataset = Dataset(x=x, y=y)
    #print(dataset.shape())  #(2,3) exemplos/linhas, features/atributos/colunas
    #print(dataset.has_label()) #ver se o dataset é supervisionado ou nao, trocar y e label por None dá Falso (nao supervisionado)
    #print(dataset.get_classes())
    #print(dataset.get_mean())
    #print(dataset.summary())
    
    x1 = np.array([[1,2,3], [1,np.nan,3], [1,2,np.nan],[8,6,9]])  #matriz
    y1 = np.array([1,2,2]) #IndexError: boolean index did not match indexed array along dimension 0; dimension is 3 but corresponding boolean dimension is 1 
    dataset2 = Dataset(x=x1, y=y1)
    print(dataset2.has_label())
    print(pd.DataFrame(x1))
    print('antes', dataset2.shape())  
    print(dataset2.fillna(4))
    #dataset2.dropna()
    print('depois',dataset2.shape())
    