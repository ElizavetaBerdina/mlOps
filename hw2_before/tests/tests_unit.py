import pickle
from swagger_api.main_models import available_models
from swagger_api.main_models import fit_model, get_preds
from swagger_api.main_models import get_params, models_storege


def test_model_classes():
    available_models_classes = list(available_models.keys())

    assert len(available_models_classes) > 0


def test_fit_model():
    model_class = list(available_models.keys())[0]
    model_name_m = 'model1'
    new_model = fit_model(model_class, model_name_m, None, [[1, 2, 3], [2, 2, 3], [3, 2, 1]], [1, 2, 3])

    assert new_model == model_name_m


def test_get_preds():
    model_class = list(available_models.keys())[0]
    model_name_m = 'model1'
    new_model = fit_model(model_class, model_name_m, None, [[1, 2, 3], [2, 2, 3], [3, 2, 1]], [1, 2, 3])
    with open(models_storege[new_model], 'rb') as f:
        model = pickle.load(f)
    target_pred_1 = model.predict([[1, 2, 3], [2, 2, 3], [3, 2, 1]])
    target_pred = get_preds(new_model, [[1, 2, 3], [2, 2, 3], [3, 2, 1]])

    assert target_pred_1 == target_pred


def test_get_params():
    model_class = list(available_models.keys())[0]
    model_name_m = 'model1'
    new_model = fit_model(model_class, model_name_m, None, [[1, 2, 3], [2, 2, 3], [3, 2, 1]], [1, 2, 3])
    with open(models_storege[new_model], 'rb') as f:
        model = pickle.load(f)
    params_1 = model.get_params()
    params_main = get_params(new_model)

    assert params_1 == params_main
