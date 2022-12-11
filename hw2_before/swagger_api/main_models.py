import pickle
import numpy as np
from sklearn.linear_model import LinearRegression
from xgboost import XGBClassifier
import os
import glob


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


model_dir = 'swagger_api/fitted_models'

if not os.path.exists(model_dir):
    os.makedirs(model_dir)

models_storege = restore_models(model_dir)

available_models = {'Regression': LinearRegression(),
                    'Classification': XGBClassifier()}


def fit_model(model_class: str, model_name: str, params: dict, train_data: np.array, train_target: np.array) -> str:
    """
    Обучает модель.
    :param model_class: класс модели.
    :type model_class: str
    :param model_name: пользовательское имя модели.
    :type model_name: str
    :param params: передаваемые параметры в модель.
    :type params: dict
    :param train_data: обучающая выборка - матрица фичей
    :type train_data: np.array
    :param train_target: таргет
    :type train_target: np.array
    :return: имя обученной модели.
    :rtype: str
    """

    if params is None:
        model = available_models[model_class]
    else:
        model = available_models[model_class].set_params(**params)
    if model is None:
        raise FileNotFoundError('Model classes not found, '
                                'Use GET model_classes')
    model.fit(train_data, train_target)
    model_name = model_name
    with open(os.path.join(model_dir, f'{model_name}.pkl'), 'wb') as f:
        pickle.dump(model, f)

    models_storege[model_name] = os.path.join(model_dir, f'{model_name}.pkl')
    return model_name


def get_preds(model_name: str, train_data: np.array) -> list:
    """
    Получает предикт предобученной модели.
    :param model_name: имя обученной модели.
    :type model_name: str
    :param train_data: обучающая выборка - матрица фичей
    :type train_data: np.array
    :return: предикт модели.
    :rtype: list
    """
    model_name_path = models_storege[model_name]
    with open(model_name_path, 'rb') as f:
        model = pickle.load(f)

    target_pred = model.predict(train_data)
    return target_pred.tolist()


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
        raise KeyError('Model not found')
