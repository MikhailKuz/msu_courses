import os
import pickle
import csv
import itertools

from collections import namedtuple
from flask_wtf import FlaskForm
from flask_bootstrap import Bootstrap
from flask import Flask, request, url_for, send_file
from flask import render_template, redirect
from flask_uploads import configure_uploads, UploadSet, UploadConfiguration

from wtforms.validators import DataRequired
from wtforms import StringField, SubmitField, validators, SelectMultipleField, IntegerField
from flask_wtf.file import FileField, FileAllowed, FileRequired
from pathvalidate import ValidationError, validate_filename

app = Flask(__name__, template_folder='html')
app.config['BOOTSTRAP_SERVE_LOCAL'] = True
app.config['SECRET_KEY'] = 'hello'
data_path = './../data/'
datasets_path = data_path +'datasets/'
models_path = data_path + 'models/'
app.config['UPLOADS_DEFAULT_DEST'] = data_path
datasets_data = UploadSet('datasets', ('csv'))
Bootstrap(app)
data_sets = os.listdir(datasets_path)
model_names = os.listdir(models_path)

configure_uploads(app, datasets_data)


def check_filename(form, field):
    try:
        validate_filename(field.data)
    except ValidationError as e:
        raise ValidationError('Wrong filename')


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

    n_estimators = IntegerField('n_estimators',
                                default=100,
                                validators=[validators.NumberRange(1,
                                                                   None,
                                                                   'Wrong n_estimators')])
    max_depth = IntegerField('max_depth', default=-1,
                             validators=[validators.NumberRange(-1,
                                                                None,
                                                                'Wrong max_depth')])

    feature_subsample_size = IntegerField('feature_subsample_size', default=-1,
                             validators=[validators.NumberRange(-1,
                                                                None,
                                                                'Wrong  feature_subsample_size')])
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
    r_f.data.choices = [(ind, data_sets[ind]) for ind in range(len(data_sets))]

    if r_f.fit.data and r_f.validate():
        print(r_f.data.data, )
        reader1, reader2 = itertools.tee(csv.reader(datasets_path + data_sets[r_f.data.data[0]],
                                                    delimiter=','))
        ncolumns = len(next(reader1))
        print(ncolumns)
        del reader1, reader2


    if menu.data.data and menu.validate():
        return redirect(url_for('datasets'))

    if menu.models.data and menu.validate():
        return redirect(url_for('models'))

    return render_template('rf.html', r_f=r_f, menu=menu)


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
