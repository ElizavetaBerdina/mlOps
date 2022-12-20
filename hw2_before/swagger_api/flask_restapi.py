from flask import jsonify, json
import flask
from flask_restx import reqparse, Resource, Api
from swagger_api.main_models import available_models
from swagger_api.main_models import fit_model, get_preds
from swagger_api.main_models import get_params, delete_model
import numpy as np
import pandas as pd
from bd_init import engine_postgres


def exist_models():
    __modelsList = pd.read_sql_query(
        """
        SELECT DISTINCT "model_name"
        FROM public.models;
        """,
        engine_postgres
    ).model_name.tolist()
    engine_postgres.dispose()

    return __modelsList


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


@api.route('/api/models')
class models_store(Resource):
    @staticmethod
    def get():
        try:
            return get_common_response(exist_models())

        except Exception as e:
            return get_error_response(e)


@api.route("/models/list")
class model_list(Resource):
    def get(self):
        __models = pd.read_sql_query(
            """
            SELECT
                "model_name" as "models_names", "model_class", "params"
            FROM public.models;
            """,
            engine_postgres
        )
        engine_postgres.dispose()
        __models.modifyDate = __models.modifyDate.astype(str)
        __models.reset_index(drop=True, inplace=True)
        __models.set_index("models", inplace=True)
        return __models.to_dict(orient="index")


fit_parser = reqparse.RequestParser()
fit_parser.add_argument("model_class",
                        type=str,
                        required=True,
                        help="Name of a class model you want to fit",
                        location="args")
fit_parser.add_argument("model_name",
                        type=str,
                        required=True,
                        help="Custom name of a model you want to fit",
                        location="args")
fit_parser.add_argument('params',
                        required=False,
                        help='Model params or None',
                        location="args")
fit_parser.add_argument('data_with_target_path',
                        type=str,
                        required=True,
                        help='Data path',
                        location="args")
# 'swagger_api/data_sample/data_regression.csv' ||
# 'swagger_api/data_sample/data_classification.csv'
# fit_fields = api.model('Fit body', {
#     'model_class': fields.String(enum=['Regression', 'Classification'],
#     description='Name of a class model'),
#     'model_name': fields.String(default = 'Regression_num_1',
#     description='Custom name of a model'),
#     'params': fields.Raw(description='Model params', required=False),
#     'data_with_target_path' : fields.String(default =
#     'swagger_api/data_sample/data_regression.csv', description='Data path')
# })


@api.route('/api/models/fit')
class models_fit(Resource):
    @api.expect(fit_parser)
#     @api.doc(body=fit_fields)
    def post(self):
        try:
            args = fit_parser.parse_args()
            if args.params is None:
                params = None
            else:
                params = json.loads(args.params.replace("'", "\""))
            path = args.data_with_target_path
            df = pd.read_csv(path, index_col=0)
            train_data = df.drop(columns=['target'])
            train_target = df['target']
            np.save('train_data.npy', train_data)
            model_name = fit_model(args.model_class, args.model_name, params, train_data, train_target)
            engine_postgres.execution_options(autocommit=True).execute(
                f"""
                              INSERT INTO public.models ("model_name", "model_classes", "params")
                              VALUES (%s,%s,%s);
                              """,
                (args.model_name, args.model_class, params)
            )
            engine_postgres.dispose()
            return get_common_response(model_name)
        except Exception as e:
            return get_error_response(e)


predict_parser = reqparse.RequestParser()
predict_parser.add_argument("model_name",
                          type=str,
                          required=True,
                          help="Custom name of a model you want to predict",
                          location="args")
# predict_fields = api.model('Predict body', {
#     'model_name': fields.String(description='Custom name of a model')
# })


@api.route('/api/models/predict')
class predict(Resource):
    @api.expect(predict_parser)
#     @api.doc(body=predict_fields)
    def post(self):
        try:
            args = predict_parser.parse_args(strict=True)
            train_data = np.load('train_data.npy')
            target_pred = get_preds(args.model_name, train_data)
            return get_common_response(target_pred)
        except Exception as e:
            return get_error_response(e)


params_parser = reqparse.RequestParser()
params_parser.add_argument("model_name",
                          type=str,
                          required=True,
                          help="Custom name of a model you want to get params",
                          location="args")
# params_fields = api.model('Params body', {
#     'model_name': fields.String(description='Custom name of a model')
# })


@api.route('/api/models/get_params')
class params(Resource):
#     @api.doc(body=params_fields)
    @api.expect(params_parser)
    def post(self):
        try:
            args = params_parser.parse_args(strict=True)
            params_mod = get_params(args.model_name)
            return get_common_response(params_mod)
        except Exception as e:
            return get_error_response(e)


delete_parser = reqparse.RequestParser()
delete_parser.add_argument('model_name',
                          type=str,
                          required=True,
                          help="Custom name of a model you want to get params",
                          location="args")

# delete_fields = api.model('Delete body', {
#     'model_name': fields.String(description='Custom name of a model')
# })


@api.route('/api/models/delete')
class delete(Resource):
#     @api.doc(body=delete_fields)
    @api.expect(delete_parser)
    def delete(self):
        try:
            args = delete_parser.parse_args(strict=True)
            delete_model(args.model_name)
            engine_postgres.execution_options(autocommit=True).execute(
                f"""
                               DELETE
                               FROM public.models
                               WHERE "model_name" = '{args.model_name}';
                               """
            )
            engine_postgres.dispose()
            return get_common_response([])
        except Exception as e:
            return get_error_response(e)


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
