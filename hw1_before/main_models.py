import pickle
import os
import glob
import numpy as np
from sklearn.datasets import make_classification, make_regression
from sklearn.linear_model import LinearRegression
from lightgbm import LGBMClassifier

def restore_models(model_dir: str) -> dict:
    """
    Список предобученных моделей.
    :param model_dir: папка с сохраненными моделями,
    относительный путь от корня запуска проекта.
    :type model_dir: str
    :return: Список предобученных моделей.
    :rtype: dict
    """
    files = glob.glob(f'./{model_dir}/*')
    models_storege = {}
    for file in files:
        models_storege[file.split('/')[-1][:-4]] = file
    return models_storege


model_dir = 'fitted_models'

if not os.path.exists(model_dir):
    os.makedirs(model_dir)
models_storege = restore_models(model_dir)

available_models = {'Regression': LinearRegression(),
                    'Classification': LGBMClassifier()}


def make_data(model_class: str):
    """
    Генерирует искуственные данные
    для обучения модели конретного класса.
    :param model_class: класс модели.
    :type model_class: str
    :return: Матрицу фичей и таргет
    :rtype:np.array
    """
    if model_class == 'Regression':
        X, y = make_regression(n_samples=200,
                               n_features=5)
    if model_class == 'Classification':
        X, y = make_classification(n_samples=200,
                                   n_features=5,
                                   n_classes=2)
    return X, y


def fit_model(model_class: str, params: dict, X: np.array, y: np.array) -> str:
    """
    Обучает модель.
    :param model_class: класс модели.
    :type model_class: str
    :param params: передаваемые параметры в модель.
    :type params: dict
    :param X: обучающая выборка - матрица фичей
    :type X: np.array
    :param y: таргет
    :type y: np.array
    :return: имя обученной модели.
    :rtype: str
    """
    model = available_models[model_class].set_params(**params)
    model.fit(X, y)
    model_name = model.__str__()
    with open(os.path.join(model_dir, f'{model_name}.pkl'), 'wb') as f:
        pickle.dump(model, f)

    models_storege[model_name] = os.path.join(model_dir, f'{model_name}.pkl')
    return model_name


def get_preds(model_name: str, X: np.array) -> list:
    """
    Получает предикт предобученной модели.
    :param model_name: имя обученной модели.
    :type model_name: str
    :param X: обучающая выборка - матрица фичей
    :type X: np.array
    :return: предикт модели.
    :rtype: list
    """
    model_name_path = models_storege[model_name]
    with open(model_name_path, 'rb') as f:
        model = pickle.load(f)

    y_pred = model.predict(X)
    return y_pred.tolist()


def get_params(model_name: str) -> dict:
    """
    Получает параметры модели.
    :param model_name: имя обученной модели.
    :type model_name: str
    :return: параметры модели.
    :rtype: dict
    """
    model_name_path = models_storege[model_name]
    with open(model_name_path, 'rb') as f:
        model = pickle.load(f)
    return model.get_params()


def delete_model(model_name: str) -> None:
    """
    Удаляет обученную модель по ее имени.
    :param model_name: имя обученной модели.
    :type model_name: str
    """
    try:
        model_path = models_storege[model_name]
        os.remove(model_path)
        del models_storege[model_name]
    except KeyError:
        raise KeyError('Данная модель не найдена')
