import os
import pickle
import csv
import itertools
import pandas as pd
from time import localtime, strftime
import sys
import json
import collections

from collections import namedtuple
from flask_wtf import FlaskForm
from flask_bootstrap import Bootstrap
from flask import Flask, request, url_for, send_file
from flask import render_template, redirect, jsonify
from flask_uploads import configure_uploads, UploadSet, UploadConfiguration

from wtforms.validators import DataRequired
from wtforms import StringField, SubmitField, validators, SelectMultipleField, IntegerField, DecimalField
from flask_wtf.file import FileField, FileAllowed, FileRequired
from pathvalidate import ValidationError, validate_filename

sys.path.insert(1, './../../../realization/')
import ensembles

app = Flask(__name__, template_folder='html')
app.config['BOOTSTRAP_SERVE_LOCAL'] = True
app.config['SECRET_KEY'] = 'hello'
data_path = './../data/'
datasets_path = os.path.join(data_path, 'datasets')
models_path = os.path.join(data_path, 'models')
info_model_path = os.path.join(data_path, 'info_models')
app.config['UPLOADS_DEFAULT_DEST'] = data_path
datasets_data = UploadSet('datasets', ('csv'))
Bootstrap(app)
data_sets = os.listdir(datasets_path)
model_names = [st[:st.rfind('.')] for st in os.listdir(models_path)]
errors = []
legend = ['feature subsample size in [-1, 0]  ->  feature_subsample_size = max columns']

configure_uploads(app, datasets_data)


def flatten(d, parent_key='', sep='_'):
    items = []
    for k, v in d.items():
        new_key = k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten(v, new_key).items())
        else:
            items.append((new_key, v))
    return dict(items)


def check_filename(form, field):
    try:
        validate_filename(field.data)
    except ValidationError as e:
        raise ValidationError('Wrong filename')


def check_feature_subsample_size(form, field):
    dataset_name = data_sets[form.data.data[0]]
    dt = pd.read_csv(os.path.join(datasets_path, dataset_name))
    if len(dt.columns) <= form.feature_subsample_size.data \
            or field.data < -1 or field.data == 0:
        del dt
        raise ValidationError('Wrong feature_subsample_size')
    del dt


def check_y(form, field):
    dataset_name = data_sets[form.data.data[0]]
    dt = pd.read_csv(os.path.join(datasets_path, dataset_name))
    if len(dt.columns) < form.y.data \
            or field.data < 0:
        del dt
        raise ValidationError('Wrong y')
    del dt


class Data(FlaskForm):
    data = SubmitField('Датасеты')


class ModelForm(FlaskForm):
    models = SubmitField('Модели')


class Menu(Data, ModelForm):
    pass


class AddData(FlaskForm):
    add_data = FileField('Добавить датасет', validators=[
        FileRequired(),
        FileAllowed(['csv'], 'csv only!')
    ])


class ModelFunc(FlaskForm):
    add_model = SubmitField('Обучить модель')
    pred_model = SubmitField('Предсказать ответ')
    info_model = SubmitField('Информация о модели')


class AddModel(FlaskForm):
    rf = SubmitField('Random Forest')
    gr = SubmitField('Gradient Boosting')


class RfForm(FlaskForm):
    name = StringField('Название модели',
                       [validators.required(), check_filename, validators.length(max=256)],
                       default='model_' + str(len(model_names)))
    data = SelectMultipleField('Датасет', validators=[DataRequired()], coerce=int)

    y = IntegerField('Номер столбца целевой переменной (начинаются от 0)',
                     default=0,
                     validators=[check_y])

    n_estimators = IntegerField('Количество деревьев',
                                default=100,
                                validators=[validators.NumberRange(1,
                                                                   None,
                                                                   'Wrong number: must be >= 1')])
    max_depth = IntegerField('Максимальная глубина дерева (-1 == max depth)', default=-1,
                             validators=[validators.NumberRange(-1,
                                                                None,
                                                                'Wrong number: must be >= -1')])

    feature_subsample_size = IntegerField('Мощность множества переборных признаков в узле (-1 == max columns)',
                                          default=-1,
                                          validators=[check_feature_subsample_size])
    fit = SubmitField('Обучить модель')


class GbForm(FlaskForm):
    name = StringField('Название модели',
                       [validators.required(), check_filename, validators.length(max=256)],
                       default='model_' + str(len(model_names)))
    data = SelectMultipleField('Датасет', validators=[DataRequired()], coerce=int)

    y = IntegerField('Номер столбца целевой переменной (начинаются от 0)',
                     default=0,
                     validators=[check_y])

    n_estimators = IntegerField('Количество деревьев',
                                default=100,
                                validators=[validators.NumberRange(1,
                                                                   None,
                                                                   'Wrong number: must be >= 1')])
    learning_rate = DecimalField('Шаг обучения',
                                 default=0.1,
                                 validators=[validators.NumberRange(0,
                                                                    None,
                                                                    'Wrong number: must be >= 0')],
                                 places=None)
    max_depth = IntegerField('Максимальная глубина дерева (-1 == max depth)', default=-1,
                             validators=[validators.NumberRange(-1,
                                                                None,
                                                                'Wrong number: must be >= -1')])

    feature_subsample_size = IntegerField('Мощность множества переборных признаков в узле (-1 == max columns)',
                                          default=-1,
                                          validators=[check_feature_subsample_size])
    fit = SubmitField('Обучить модель')


class PredForm(FlaskForm):
    model = SelectMultipleField('Модель', validators=[DataRequired()], coerce=int)
    data = SelectMultipleField('Датасет', validators=[DataRequired()], coerce=int)
    y = IntegerField('Номер столбца целевой переменной (начинаются от 0)',
                     default=0,
                     validators=[check_y])
    predict = SubmitField('Загрузить ответ')


class InfoForm(FlaskForm):
    model = SelectMultipleField('Модель', validators=[DataRequired()], coerce=int)
    download = SubmitField('Загрузить параметры')
    print_par = SubmitField('Распечатать параметры')


@app.route('/', methods=['GET', 'POST'])
@app.route('/index', methods=['GET', 'POST'])
def index():
    menu = Menu()

    if menu.data.data and menu.validate():
        return redirect(url_for('datasets'))
    if menu.models.data and menu.validate():
        return redirect(url_for('models'))
    return render_template('index.html', menu=menu)


@app.route('/datasets', methods=['GET', 'POST'])
def datasets():
    mdls = ModelForm()
    add_data = AddData()
    try:
        if add_data.add_data.data and add_data.validate():
            filename = datasets_data.save(add_data.add_data.data)
            if filename not in data_sets:
                data_sets.append(filename)
        if mdls.models.data and mdls.validate():
            return redirect(url_for('models'))

        return render_template('datasets.html', add_data=add_data, data_sets=data_sets, mdls=mdls)
    except Exception as exc:
        app.logger.info('Exception: {0}'.format(exc))
    return render_template('datasets.html', add_data=add_data, data_sets=data_sets, mdls=mdls)


@app.route('/models', methods=['GET', 'POST'])
def models():
    data = Data()
    model_func = ModelFunc()
    errors = []
    try:
        if model_func.add_model.data and model_func.validate():
            return redirect(url_for('addmodel'))

        if model_func.pred_model.data and model_func.validate():
            return redirect(url_for('predmodel'))

        if model_func.info_model.data and model_func.validate():
            return redirect(url_for('infomodel'))

        if data.data.data and data.validate():
            return redirect(url_for('datasets'))
    except Exception as exc:
        app.logger.info('Exception: {0}'.format(exc))
    return render_template('models.html', data=data, model_func=model_func, model_names=model_names)


@app.route('/infomodel', methods=['GET', 'POST'])
def infomodel():
    menu = Menu()
    info = InfoForm()
    js = {}
    info.model.choices = [(ind, model_names[ind]) for ind in range(len(model_names))]

    try:
        if info.download.data:
            model_name = model_names[info.model.data[0]]
            info_path = os.path.join(info_model_path, model_name + '.json')
            return send_file(info_path, as_attachment=True)

        if info.print_par.data:
            model_name = model_names[info.model.data[0]]
            info_path = os.path.join(info_model_path, model_name + '.json')
            js = json.load(open(info_path, "r"))
            js = flatten(js)
            scores = js.pop('scores', None)
            time = js.pop('time', None)
            return render_template('infomodel.html', menu=menu, info=info, js=js, image=image)

        if menu.data.data and menu.validate():
            return redirect(url_for('datasets'))

        if menu.models.data and menu.validate():
            return redirect(url_for('models'))
    except Exception as exc:
        app.logger.info('Exception: {0}'.format(exc))
    return render_template('infomodel.html', menu=menu, info=info, errors=errors)


@app.route('/predmodel', methods=['GET', 'POST'])
def predmodel():
    menu = Menu()
    pred = PredForm()
    pred.data.choices = [(ind, data_sets[ind]) for ind in range(len(data_sets))]
    pred.model.choices = [(ind, model_names[ind]) for ind in range(len(model_names))]

    try:
        if pred.predict.data:
            if not pred.validate():
                return render_template('predmodel.html', menu=menu, pred=pred, errors=errors)
            dataset_name = data_sets[pred.data.data[0]]
            model_name = model_names[pred.model.data[0]]
            ans_path = os.path.join(data_path, 'answer.csv')
            dt = pd.read_csv(os.path.join(datasets_path, dataset_name))
            y_name = dt.columns[pred.y.data]
            X = dt.drop(columns=[y_name]).values
            model = pickle.load(open(os.path.join(models_path, model_name + ".pkl"), "rb"))
            try:
                preds = model.predict(X)
            except Exception as e:
                errors.append(('Wrong dataset format', strftime("%Y-%m-%d %H:%M:%S", localtime())))
                app.logger.info('Exception: {0}'.format(e))
                return render_template('predmodel.html', menu=menu, pred=pred, errors=errors)
            finally:
                del dt

            ans = pd.DataFrame(columns=['target'], data=preds, index=list(range(len(preds))))
            ans.to_csv(ans_path)
            return send_file(ans_path, as_attachment=True)

        if menu.data.data and menu.validate():
            return redirect(url_for('datasets'))

        if menu.models.data and menu.validate():
            return redirect(url_for('models'))
    except Exception as exc:
        app.logger.info('Exception: {0}'.format(exc))
    return render_template('predmodel.html', menu=menu, pred=pred, errors=errors)


@app.route('/addmodel', methods=['GET', 'POST'])
def addmodel():
    menu = Menu()
    addmodel = AddModel()

    if addmodel.rf.data and addmodel.validate():
        errors.clear()
        return redirect(url_for('rf'))

    if addmodel.gr.data and addmodel.validate():
        return redirect(url_for('grad'))

    if menu.data.data and menu.validate():
        return redirect(url_for('datasets'))

    if menu.models.data and menu.validate():
        return redirect(url_for('models'))

    return render_template('addmodel.html', addmodel=addmodel, menu=menu)


@app.route('/rf', methods=['GET', 'POST'])
def rf():
    menu = Menu()
    r_f = RfForm()
    r_f.data.choices = [(ind, data_sets[ind]) for ind in range(len(data_sets))]

    try:
        if r_f.fit.data:
            if not r_f.validate():
                return render_template('rf.html', r_f=r_f, menu=menu, errors=errors)
            dataset_name = data_sets[r_f.data.data[0]]
            dt = pd.read_csv(os.path.join(datasets_path, dataset_name))
            name = r_f.name.data
            y_name = dt.columns[r_f.y.data]
            n_estimators = r_f.n_estimators.data
            max_depth = r_f.max_depth.data
            feature_subsample_size = r_f.feature_subsample_size.data
            X, y = dt.drop(columns=[y_name]).values, dt[y_name].values
            if max_depth == -1:
                max_depth = None
            if feature_subsample_size == -1:
                feature_subsample_size = None

            rf_model = ensembles.RandomForestMSE(n_estimators, max_depth, feature_subsample_size)
            try:
                info = rf_model.fit(X, y, X, y)
            except Exception as e:
                errors.append(('Wrong dataset format', strftime("%Y-%m-%d %H:%M:%S", localtime())))
                app.logger.info('Exception: {0}'.format(e))
                return render_template('rf.html', r_f=r_f, menu=menu, errors=errors)
            finally:
                del dt
            allinfo = {
                'Тренировочный датасет': dataset_name,
                'params': {
                    'Алгоритм': 'random forest',
                    r_f.n_estimators.label.text: n_estimators,
                    r_f.max_depth.label.text: max_depth,
                    r_f.feature_subsample_size.label.text: feature_subsample_size
                },
                'loss info': info
            }
            json.dump(allinfo, open(os.path.join(data_path, 'info_models', name + '.json'), 'w'), indent=4)
            pickle.dump(rf_model, open(os.path.join(data_path, 'models', name + '.pkl'), 'wb'))
            if name not in model_names:
                model_names.append(name)
            return redirect(url_for('models'))

        if menu.data.data and menu.validate():
            return redirect(url_for('datasets'))

        if menu.models.data and menu.validate():
            return redirect(url_for('models'))
    except Exception as exc:
        app.logger.info('Exception: {0}'.format(exc))
    return render_template('rf.html', r_f=r_f, menu=menu, errors=errors)


@app.route('/grad', methods=['GET', 'POST'])
def grad():
    menu = Menu()
    g_d = GbForm()
    g_d.data.choices = [(ind, data_sets[ind]) for ind in range(len(data_sets))]

    try:
        if g_d.fit.data:
            if not g_d.validate():
                return render_template('grad.html', g_d=g_d, menu=menu, errors=errors)
            dataset_name = data_sets[g_d.data.data[0]]
            dt = pd.read_csv(datasets_path + dataset_name)
            name = g_d.name.data
            y_name = dt.columns[g_d.y.data]
            n_estimators = g_d.n_estimators.data
            learning_rate = float(g_d.learning_rate.data)
            max_depth = g_d.max_depth.data
            feature_subsample_size = g_d.feature_subsample_size.data
            X, y = dt.drop(columns=[y_name]).values, dt[y_name].values
            if max_depth == -1:
                max_depth = None
            if feature_subsample_size == -1:
                feature_subsample_size = None

            gd_model = ensembles.GradientBoostingMSE(n_estimators, learning_rate, max_depth, feature_subsample_size)
            try:
                info = gd_model.fit(X, y, X, y)
            except Exception as e:
                errors.append(('Wrong dataset format', strftime("%Y-%m-%d %H:%M:%S", localtime())))
                app.logger.info('Exception: {0}'.format(e))
                return render_template('grad.html', r_f=g_d, menu=menu, errors=errors)
            finally:
                del dt
            allinfo = {
                'Тренировочный датасет': dataset_name,
                'params': {
                    'Алгоритм': 'gradient boosting',
                    g_d.n_estimators.label.text: n_estimators,
                    g_d.learning_rate.label.text: learning_rate,
                    g_d.max_depth.label.text: max_depth,
                    g_d.feature_subsample_size.label.text: feature_subsample_size
                },
                'loss info': info
            }
            json.dump(allinfo, open(os.path.join(data_path, 'info_models', name + '.json'), 'w'), indent=4)
            pickle.dump(gd_model, open(os.path.join(data_path, 'models', name + '.pkl'), 'wb'))
            if name not in model_names:
                model_names.append(name)
            return redirect(url_for('models'))

        if menu.data.data and menu.validate():
            return redirect(url_for('datasets'))

        if menu.models.data and menu.validate():
            return redirect(url_for('models'))
    except Exception as exc:
        app.logger.info('Exception: {0}'.format(exc))
    return render_template('grad.html', g_d=g_d, menu=menu, errors=errors)


@app.route('/clear_messages', methods=['POST'])
def clear_messages():
    messages.clear()
    return redirect(url_for('prepare_message'))

