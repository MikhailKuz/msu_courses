import os
import pickle
import csv
import itertools
import pandas as pd
from time import localtime, strftime
import sys
import json
import collections
import matplotlib.pyplot as plt
import seaborn as sns

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
from matplotlib.ticker import MaxNLocator

sys.path.insert(1, './')
import ensembles

app = Flask(__name__,
            template_folder='html',
            static_url_path='',
            static_folder='graphics')
app.config['BOOTSTRAP_SERVE_LOCAL'] = True
app.config['SECRET_KEY'] = 'hello'
data_path = '../data/'
datasets_path = os.path.join(data_path, 'datasets')
models_path = os.path.join(data_path, 'models')
info_model_path = os.path.join(data_path, 'info_models')
img_path_s = os.path.join('.', 'graphics', 'plot.jpg')
app.config['UPLOADS_DEFAULT_DEST'] = data_path
datasets_data = UploadSet('datasets', ('csv'))

Bootstrap(app)
data_sets = os.listdir(datasets_path)
model_names = [st[:st.rfind('.')] for st in os.listdir(models_path)]
errors = []
dataset_name_cur = ''
ncolumns_cur = 0

sns.set_context("talk")
sns.set_style('darkgrid')
plt.rcParams.update({'font.size': 18})
plt.rcParams['legend.title_fontsize'] = 'xx-small'
sns.set_theme('poster')
title_leg_size = '15'
font_leg_size = '15'
img = False

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


def check_ncolumns(form):
    global ncolumns_cur
    global dataset_name_cur
    dataset_name = data_sets[form.data.data[0]]
    if dataset_name != dataset_name_cur:
        dt = pd.read_csv(os.path.join(datasets_path, dataset_name))
        ncolumns_cur = len(dt.columns)
        dataset_name_cur = dataset_name


def str_to_ints(string):
    if string == '':
        return []
    st = [x.strip() for x in string.split(',')]
    integers = [int(x) for x in st]
    return integers


def check_filename(form, field):
    try:
        validate_filename(field.data)
    except ValidationError as e:
        raise ValidationError('Wrong filename')


def check_feature_subsample_size(form, field):
    check_ncolumns(form)
    if ncolumns_cur < form.feature_subsample_size.data \
            or field.data < -1 or field.data == 0:
        raise ValidationError('Wrong feature_subsample_size')


def check_fit_y(form, field):
    check_ncolumns(form)
    if ncolumns_cur <= form.y.data \
            or field.data < 0:
        raise ValidationError('Wrong y')


def check_pred_y(form, field):
    check_ncolumns(form)
    if ncolumns_cur <= form.y.data \
            or field.data < -1:
        raise ValidationError('Wrong y')


def check_drop_col(form, field):
    check_ncolumns(form)
    try:
        integers = str_to_ints(field.data)
        for integer in integers:
            if integer >= ncolumns_cur \
                    or integer < 0:
                raise Exception
    except Exception as e:
        raise ValidationError('Wrong drop columns')


class Data(FlaskForm):
    data = SubmitField('Datasets')


class ModelForm(FlaskForm):
    models = SubmitField('Models')


class Menu(Data, ModelForm):
    pass


class GeneralPar(FlaskForm):
    name = StringField("Model's name",
                       [validators.required(), check_filename, validators.length(max=256)],
                       default='model_' + str(len(model_names)))
    data = SelectMultipleField('Dataset', validators=[DataRequired()], coerce=int)

    drop_col = StringField("Drop column's numbers (begins with 0, comma separated)",
                           [check_drop_col],
                           default='')

    y = IntegerField("Target column's number (begins with 0)",
                     default=0,
                     validators=[check_fit_y])

    n_estimators = IntegerField('Number of trees',
                                default=100,
                                validators=[validators.NumberRange(1,
                                                                   None,
                                                                   'Wrong number: must be >= 1')])


class AddData(FlaskForm):
    add_data = FileField('Add dataset', validators=[
        FileRequired(),
        FileAllowed(['csv'], 'csv only!')
    ])


class ModelFunc(FlaskForm):
    add_model = SubmitField('Train model')
    pred_model = SubmitField('Predict answer')
    info_model = SubmitField("Model's information")


class AddModel(FlaskForm):
    rf = SubmitField('Random Forest')
    gr = SubmitField('Gradient Boosting')


class RfForm(GeneralPar):
    max_depth = IntegerField('Max depth of tree (-1 == max depth)', default=-1,
                             validators=[validators.NumberRange(-1,
                                                                None,
                                                                'Wrong number: must be >= -1')])

    feature_subsample_size = IntegerField('Feature subsample size (-1 == max columns)',
                                          default=-1,
                                          validators=[check_feature_subsample_size])
    fit = SubmitField('Train model')


class GbForm(GeneralPar):
    learning_rate = DecimalField('Learning rate',
                                 default=0.1,
                                 validators=[validators.NumberRange(0,
                                                                    None,
                                                                    'Wrong number: must be >= 0')],
                                 places=None)
    max_depth = IntegerField('Max depth of tree (-1 == max depth)', default=5,
                             validators=[validators.NumberRange(-1,
                                                                None,
                                                                'Wrong number: must be >= -1')])

    feature_subsample_size = IntegerField('Feature subsample size (-1 == max columns)',
                                          default=-1,
                                          validators=[check_feature_subsample_size])
    fit = SubmitField('Train model')


class PredForm(FlaskForm):
    model = SelectMultipleField('Model', validators=[DataRequired()], coerce=int)
    data = SelectMultipleField('Dataset', validators=[DataRequired()], coerce=int)
    drop_col = StringField("Drop column's numbers (begins with 0, comma separated",
                           [check_drop_col],
                           default='0')
    predict = SubmitField('Download answer')


class InfoForm(FlaskForm):
    model = SelectMultipleField('Model', validators=[DataRequired()], coerce=int)
    download = SubmitField('Download parametrs')
    print_par = SubmitField('Print parametrs')


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
    global errors
    data = Data()
    model_func = ModelFunc()
    errors = []
    try:
        if model_func.add_model.data and model_func.validate():
            return redirect(url_for('addmodel'))

        if model_func.pred_model.data and model_func.validate():
            return redirect(url_for('predmodel'))

        if model_func.info_model.data and model_func.validate():
            global img
            img = False
            return redirect(url_for('infomodel'))

        if data.data.data and data.validate():
            return redirect(url_for('datasets'))
    except Exception as exc:
        app.logger.info('Exception: {0}'.format(exc))
    return render_template('models.html', data=data, model_func=model_func, model_names=model_names)


@app.route('/infomodel', methods=['GET', 'POST'])
def infomodel():
    global img
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

            h = [dict([('scores', scores), ('time', time)])]
            lab_v = [js['Алгоритм']]

            n = 0
            fig, axs = plt.subplots(figsize=(2 * 10, 5), ncols=1, nrows=1)
            p = []
            ticks = list(range(1, len(h[0]['scores']) + 1))
            for history in h:
                if len(ticks[n:]) > 1:
                    p.append(axs.plot(ticks[n:], history['scores'][n:], linewidth=5))
                else:
                    p.append(axs.plot(ticks[n:], history['scores'][n:], marker='o', markersize=10))

            axs.set_title('Dependence of quality on number of trees')
            axs.set_ylabel('rmse')
            axs.set_xlabel('Number of trees')
            axs.xaxis.set_major_locator(MaxNLocator(integer=True))
            axs.grid(True)

            leg = axs.legend((pj[0] for pj in p), lab_v)
            leg.set_title('algorithm')
            plt.setp(axs.get_legend().get_texts(), fontsize=font_leg_size)
            plt.setp(axs.get_legend().get_title(), fontsize=title_leg_size)
            fig.savefig(img_path_s, bbox_inches='tight')
            img = True
            return render_template('infomodel.html', menu=menu, info=info, js=js, img=img)

        if menu.data.data and menu.validate():
            return redirect(url_for('datasets'))

        if menu.models.data and menu.validate():
            return redirect(url_for('models'))
    except Exception as exc:
        app.logger.info('Exception: {0}'.format(exc))
    return render_template('infomodel.html', menu=menu, info=info, js=js, img=img)


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
            drop_col = list(dt.columns[str_to_ints(pred.drop_col.data)])
            X = dt.drop(columns=drop_col).values
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
            drop_col = list(dt.columns[str_to_ints(r_f.drop_col.data)])
            if y_name in drop_col:
                errors.append(('Target variable contains in drop columns set',
                               strftime("%Y-%m-%d %H:%M:%S", localtime())))
                del dt
                return render_template('grad.html', r_f=r_f, menu=menu, errors=errors)
            X, y = dt.drop(columns=[y_name] + drop_col).values, dt[y_name].values
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
                'Training dataset': dataset_name,
                'params': {
                    'Algorithm': 'random forest',
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
            dt = pd.read_csv(os.path.join(datasets_path, dataset_name))
            name = g_d.name.data
            y_name = dt.columns[g_d.y.data]
            n_estimators = g_d.n_estimators.data
            learning_rate = float(g_d.learning_rate.data)
            max_depth = g_d.max_depth.data
            feature_subsample_size = g_d.feature_subsample_size.data
            drop_col = list(dt.columns[str_to_ints(g_d.drop_col.data)])
            if y_name in drop_col:
                errors.append(('Target variable contains in drop columns set',
                               strftime("%Y-%m-%d %H:%M:%S", localtime())))
                del dt
                return render_template('grad.html', g_d=g_d, menu=menu, errors=errors)
            X, y = dt.drop(columns=[y_name] + drop_col).values, dt[y_name].values
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
                return render_template('grad.html', g_d=g_d, menu=menu, errors=errors)
            finally:
                del dt
            allinfo = {
                'Training dataset': dataset_name,
                'params': {
                    'Algorithm': 'gradient boosting',
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
