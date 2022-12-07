from flask import jsonify, json
import flask
from flask_restx import reqparse, Resource, Api
from main_models import make_data, models_storege, available_models
from main_models import fit_model, get_preds, get_params, delete_model
import numpy as np


class HttpStatus:
    OK = 200
    CREATED = 201
    NOT_FOUND = 404
    BAD_REQUEST = 400


def get_error_response(exception):
    construct = {
        'error': exception.__str__(),
        'success': False,
        'result': []
    }
    response = jsonify(construct)
    response.status_code = HttpStatus.OK
    return response


def get_common_response(result):
    construct = {
        'error': [],
        'success': True,
        'result': result
    }
    response = jsonify(construct)
    response.status_code = HttpStatus.OK
    return response


def fix_list(arr):
    return [eval(item) for item in arr]


app = flask.Flask(__name__)
api = Api(app)
available_models_classes = list(available_models.keys())


@api.route('/api/model_classes')
class model_classes(Resource):
    @staticmethod
    def get():
        try:
            return get_common_response(available_models_classes)
        except Exception as e:
            return get_error_response(e)


fit_parser = reqparse.RequestParser()
fit_parser.add_argument('model_class')
fit_parser.add_argument('params', help='(необязательное поле)')


@api.route('/api/models/fit')
class models_fit(Resource):
    @staticmethod
    def get():
        try:
            return get_common_response(list(models_storege.keys()))

        except Exception as e:
            return get_error_response(e)

    @api.expect(fit_parser)
    def post(self):
        try:
            args = fit_parser.parse_args()
            if args.params is None:
                params = None
            else:
                params = json.loads(args.params.replace("'", "\""))
            X, y = make_data(args.model_class)
            np.save('X', X)
            model_name = fit_model(args.model_class, params, X, y)
            return get_common_response(model_name)
        except Exception as e:
            return get_error_response(e)


predict_parser = reqparse.RequestParser()
predict_parser.add_argument('model_name')


@api.route('/api/models/predict')
class predict(Resource):
    @api.expect(predict_parser)
    def post(self):
        try:
            args = predict_parser.parse_args()
            X = np.load('X.npy')
            y_pred = get_preds(args.model_name, X)
            return get_common_response(y_pred)
        except Exception as e:
            return get_error_response(e)


params_parser = reqparse.RequestParser()
params_parser.add_argument('model_name')


@api.route('/api/models/get_params')
class params(Resource):
    @api.expect(params_parser)
    def post(self):
        try:
            args = params_parser.parse_args()
            params_mod = get_params(args.model_name)
            return get_common_response(params_mod)
        except Exception as e:
            return get_error_response(e)


delete_parser = reqparse.RequestParser()
delete_parser.add_argument('model_name')


@api.route('/api/models/delete')
class delete(Resource):
    @api.expect(delete_parser)
    def delete(self):
        try:
            args = delete_parser.parse_args()
            delete_model(args.model_name)
            return get_common_response([])
        except Exception as e:
            return get_error_response(e)
