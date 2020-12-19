import os
import pickle
import csv
import itertools
import pandas as pd
from time import localtime, strftime
import sys

from collections import namedtuple
from flask_wtf import FlaskForm
from flask_bootstrap import Bootstrap
from flask import Flask, request, url_for, send_file
from flask import render_template, redirect
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
datasets_path = data_path + 'datasets/'
models_path = data_path + 'models/'
app.config['UPLOADS_DEFAULT_DEST'] = data_path
datasets_data = UploadSet('datasets', ('csv'))
Bootstrap(app)
data_sets = os.listdir(datasets_path)
model_names = [st[:st.rfind('.')] for st in os.listdir(models_path)]
errors = []
legend = ['feature subsample size in [-1, 0]  ->  feature_subsample_size = max columns']

configure_uploads(app, datasets_data)


def check_filename(form, field):
    try:
        validate_filename(field.data)
    except ValidationError as e:
        raise ValidationError('Wrong filename')


def check_feature_subsample_size(form, field):
    dataset_name = data_sets[form.data.data[0]]
    dt = pd.read_csv(datasets_path + dataset_name)
    if len(dt.columns) <= form.feature_subsample_size.data \
            or field.data < -1 or field.data == 0:
        del dt
        raise ValidationError('Wrong feature_subsample_size')
    del dt


def check_y(form, field):
    dataset_name = data_sets[form.data.data[0]]
    dt = pd.read_csv(datasets_path + dataset_name)
    if len(dt.columns) < form.feature_subsample_size.data \
            or field.data < 0:
        del dt
        raise ValidationError('Wrong y')
    del dt


class Message:
    text = ''


class TextForm(FlaskForm):
    text = StringField('Text', validators=[DataRequired()])
    submit = SubmitField('Get Result')


class Response(FlaskForm):
    score = StringField('Score', validators=[DataRequired()])
    sentiment = StringField('Sentiment', validators=[DataRequired()])
    submit = SubmitField('Try Again')


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
    add_model = SubmitField('Добавить модель')
    pred_model = SubmitField('Предсказать ответ')
    info_model = SubmitField('Информация о модели')


class AddModel(FlaskForm):
    rf = SubmitField('Random Forest')
    gr = SubmitField('Gradient Boosting')


class Rf(FlaskForm):
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


class Gb(FlaskForm):
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


def score_text(text):
    try:
        model = pickle.load(open(os.path.join(data_path, "logreg.pkl"), "rb"))
        tfidf = pickle.load(open(os.path.join(data_path, "tf-idf.pkl"), "rb"))

        score = model.predict_proba(tfidf.transform([text]))[0][1]
        sentiment = 'positive' if score > 0.5 else 'negative'
    except Exception as exc:
        app.logger.info('Exception: {0}'.format(exc))
        score, sentiment = 0.0, 'unknown'

    return score, sentiment


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
            data_sets.append(filename)
        if mdls.models.data and mdls.validate():
            return redirect(url_for('models'))

        return render_template('datasets.html', add_data=add_data, data_sets=data_sets, mdls=mdls)
    except Exception as exc:
        app.logger.info('Exception: {0}'.format(exc))
    return redirect(url_for('datasets'))


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

        return render_template('models.html', data=data, model_func=model_func, model_names=model_names)
    except Exception as exc:
        app.logger.info('Exception: {0}'.format(exc))
    return redirect(url_for('models'))


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
    r_f = Rf()
    r_f.data.choices = [(ind, data_sets[ind]) for ind in range(len(data_sets) - 1, -1, -1)]

    try:
        # if not r_f.validate():
        #     return render_template('rf.html', r_f=r_f, menu=menu, errors=errors)

        if r_f.fit.data and r_f.validate():
            dataset_name = data_sets[r_f.data.data[0]]
            dt = pd.read_csv(datasets_path + dataset_name)
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
                'training dataset': dataset_name,
                'params': {
                    'algorith': 'random forest',
                    r_f.n_estimators.label: n_estimators,
                    r_f.max_depth.label: max_depth,
                    r_f.feature_subsample_size.label: feature_subsample_size
                },
                'loss info': info
            }
            pickle.dump(allinfo, open(os.path.join(data_path, 'info_models', name + '.pkl'), 'wb'))
            pickle.dump(rf_model, open(os.path.join(data_path, 'models', name + '.pkl'), 'wb'))
            model_names.append(name)
            return redirect(url_for('models'))

        # if menu.data.data and menu.validate():        # срабатывает при нажатии на r_f.fit
        #     return redirect(url_for('datasets'))
        #
        # if menu.models.data and menu.validate():
        #     return redirect(url_for('models'))
    except Exception as exc:
        app.logger.info('Exception: {0}'.format(exc))
    return render_template('rf.html', r_f=r_f, menu=menu, errors=errors)


@app.route('/grad', methods=['GET', 'POST'])
def grad():
    menu = Menu()
    g_d = Gb()
    g_d.data.choices = [(ind, data_sets[ind]) for ind in range(len(data_sets) - 1, -1, -1)]

    try:
        # if not r_f.validate():
        #     return render_template('rf.html', r_f=r_f, menu=menu, errors=errors)

        if g_d.fit.data and g_d.validate():
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
                'training dataset': dataset_name,
                'params': {
                    'algorith': 'gradient boosting',
                    g_d.n_estimators.label: n_estimators,
                    g_d.learning_rate.label: learning_rate,
                    g_d.max_depth.label: max_depth,
                    g_d.feature_subsample_size.label: feature_subsample_size
                },
                'loss info': info
            }
            pickle.dump(allinfo, open(os.path.join(data_path, 'info_models', name + '.pkl'), 'wb'))
            pickle.dump(gd_model, open(os.path.join(data_path, 'models', name + '.pkl'), 'wb'))
            model_names.append(name)
            return redirect(url_for('models'))

        # if menu.data.data and menu.validate():        # срабатывает при нажатии на r_f.fit
        #     return redirect(url_for('datasets'))
        #
        # if menu.models.data and menu.validate():
        #     return redirect(url_for('models'))
    except Exception as exc:
        app.logger.info('Exception: {0}'.format(exc))
    return render_template('grad.html', r_f=g_d, menu=menu, errors=errors)


@app.route('/clear_messages', methods=['POST'])
def clear_messages():
    messages.clear()
    return redirect(url_for('prepare_message'))


@app.route('/messages', methods=['GET', 'POST'])
def prepare_message():
    message = Message()

    if request.method == 'POST':
        message.header, message.text = request.form['header'], request.form['text']
        messages.append(message)

        return redirect(url_for('prepare_message'))

    return render_template('messages.html', messages=messages)


@app.route('/result', methods=['GET', 'POST'])
def get_result():
    try:
        response_form = Response()

        if response_form.validate_on_submit():
            return redirect(url_for('get_text_score'))

        score = request.args.get('score')
        sentiment = request.args.get('sentiment')

        response_form.score.data = score
        response_form.sentiment.data = sentiment

        return render_template('from_form.html', form=response_form)
    except Exception as exc:
        app.logger.info('Exception: {0}'.format(exc))


@app.route('/sentiment', methods=['GET', 'POST'])
def get_text_score():
    try:
        text_form = TextForm()

        if text_form.validate_on_submit():
            app.logger.info('On text: {0}'.format(text_form.text.data))
            score, sentiment = score_text(text_form.text.data)
            app.logger.info("Score: {0:.3f}, Sentiment: {1}".format(score, sentiment))
            text_form.text.data = ''
            return redirect(url_for('get_result', score=score, sentiment=sentiment))
        return render_template('from_form.html', form=text_form)
    except Exception as exc:
        app.logger.info('Exception: {0}'.format(exc))
