import pandas as pd
from dash import Dash, dcc, html, callback, Input, Output, State, dash_table
import numpy as np
import openpyxl
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn import tree
import datetime as dt

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from xgboost import XGBClassifier

from optuna import Trial, create_study, visualization
from optuna.samplers import TPESampler
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from keras.layers import Input as InputLayer
from keras.layers import Dense
from keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import SGD

app = Dash(__name__)

people = pd.read_csv('people.csv').head(10000)
activites_train = pd.read_csv('act_train.csv').head(10000)
activites_test = pd.read_csv('act_test.csv').head(3000)

def merge_sets(people, activities):
    return activities.join(people.set_index('people_id'), on='people_id', lsuffix='_activity', rsuffix='_person')

def standardization(df,col):
    std_scaler = MinMaxScaler()
    for i in col:
        arr = df[i]
        arr = np.array(arr)
        df[i] = std_scaler.fit_transform(arr.reshape(len(arr),1))
    return df

def preprocess(dataset):
    dataset = dataset.drop(['date_person', 'date_activity', 'people_id', 'activity_id'], axis=1) # 
    null_percentage = dataset.isnull().sum()/dataset.shape[0]*100
    
    col_to_drop = null_percentage[null_percentage>60].keys()

    output_dataset = dataset.drop(col_to_drop, axis=1)
    num_cols = output_dataset._get_numeric_data().columns
    categorical_cols = list(set(output_dataset.columns)-set(num_cols))
    numeric_col = output_dataset.select_dtypes(include='number').columns
    for categorical_column in categorical_cols:
        print('Info ', categorical_column, output_dataset[categorical_column].unique())
    listed_num_cols = num_cols.tolist()
    if "outcome" in listed_num_cols:
        num_cols = num_cols.drop('outcome')
    output_dataset = standardization(output_dataset,num_cols)
    output_dataset = pd.get_dummies(output_dataset,columns=categorical_cols,prefix="",prefix_sep="")

    return output_dataset

def main_logic(passed_people_df=people, passed_activities_df=activites_train):
    original = merge_sets(passed_people_df, passed_activities_df)
    original_preprocessed = preprocess(original)
    X = original_preprocessed.loc[:, original_preprocessed.columns != 'outcome']
    y = original_preprocessed['outcome']
    x_train, x_val, y_train, y_val = train_test_split(X, y, test_size=0.33, stratify=y, random_state=42)
    return x_train, x_val, y_train, y_val, X, original

def score_logic(y_real, predictions):
        accuracy = accuracy_score(y_real, predictions)
        f1 = f1_score(y_real, predictions, average='micro')
        precision = precision_score(y_real, predictions, average='micro')
        recall = recall_score(y_real, predictions, average='micro')
        return round(accuracy,3), round(f1, 3), round(precision, 3), round(recall, 3)
    
def decision_logic(max_depth=6, max_leaves=0, with_boosting=False, min_samples_leaf=5, people_df=people, activities_df=activites_train):
        x_train, x_val, y_train, y_val, X, original = main_logic(people_df, activities_df)
        if(with_boosting):
            model = XGBClassifier(objective='binary:logistic', min_samples_leaf=min_samples_leaf, use_label_encoder=False, verbosity=2, max_depth=max_depth, max_leaves=max_leaves)
            model.fit(x_train.values, y_train.values)
            model.score(x_val.values, y_val.values)
            predictions = model.predict(x_val)
            accuracy, f1, precision, recall = score_logic(y_val, predictions)
            predictions_main = model.predict(X)
            null_percentage = original.isnull().sum()/original.shape[0]*100
            col_to_drop = null_percentage[null_percentage>60].keys()
            original = original.drop(col_to_drop, axis=1)
            original['predictions'] = predictions_main
            return accuracy, f1, precision, recall, original
        if(with_boosting == False):
            clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=max_depth, min_samples_leaf=min_samples_leaf, random_state=100)
            clf = clf.fit(x_train.values, y_train.values)
            predictions = clf.predict(x_val)
            accuracy, f1, precision, recall = score_logic(y_val, predictions)
            predictions_main = clf.predict(X)
            null_percentage = original.isnull().sum()/original.shape[0]*100
            col_to_drop = null_percentage[null_percentage>60].keys()
            original = original.drop(col_to_drop, axis=1)
            original['predictions'] = predictions_main
            original['predictions'] = predictions_main
            return accuracy, f1, precision, recall, original

def logistic_regression_logic(regularization='l1', regularization_coeff=0.9, people_df=people, activities_df=activites_train):
    x_train, x_val, y_train, y_val, X, original = main_logic(people_df, activities_df)
    logisticRegr = LogisticRegression(penalty=regularization, C=regularization_coeff, solver='liblinear')
    logisticRegr.fit(x_train, y_train)
    predictions = logisticRegr.predict(x_val)
    score = logisticRegr.score(x_val, y_val)
    accuracy, f1, precision, recall = score_logic(y_val, predictions)
    predictions_main = logisticRegr.predict(X)
    null_percentage = original.isnull().sum()/original.shape[0]*100
    col_to_drop = null_percentage[null_percentage>60].keys()
    original = original.drop(col_to_drop, axis=1)
    original['predictions'] = predictions_main
    return accuracy, f1, precision, recall, original

def neural_network_logic(optimizer='SGD', learning_rate=0.1, layers_number=3, logits_for_layers=64, activation_func='relu', people_df=people, activities_df=activites_train):
    x_train, x_val, y_train, y_val, X, original  = main_logic(people_df, activities_df)
    model = Sequential()
    model.add(InputLayer(1113))
    for layer in range(1, layers_number):
        model.add(Dense(logits_for_layers, activation=activation_func))
    model.add(Dense(1, activation='sigmoid'))
    opt = ''
    if(optimizer == 'Adam'):
        opt = keras.optimizers.Adam(learning_rate=learning_rate)
    if(optimizer == 'SGD'):
        opt = keras.optimizers.SGD(learning_rate=learning_rate)
    model.compile(loss='mean_squared_error', optimizer=opt)
    model.fit(np.asarray(x_train).astype(np.float32), y_train, epochs=15, batch_size=100)
    pred_binary = np.argmax(np.round(np.asarray(x_val).astype(np.float32),0), axis=1)
    score = np.sqrt(mean_squared_error(pred_binary, y_val))
    accuracy, f1, precision, recall = score_logic(y_val, pred_binary)
    predictions_main = model.predict(X)
    null_percentage = original.isnull().sum()/original.shape[0]*100
    col_to_drop = null_percentage[null_percentage>60].keys()
    original = original.drop(col_to_drop, axis=1)
    original['predictions'] = np.argmax(np.round(np.asarray(predictions_main).astype(np.float32),0), axis=1)
    return accuracy, f1, precision, recall, original

def render_datetime(data, y_label, x_label, img_name, cell_num, wb, title, file_name):
    fig, ax = plt.subplots(figsize=(9,3))
    sns.lineplot(data=data, x='date_activity', y='activity_id', ax=ax)
    ax.set_ylabel(y_label); ax.set_xlabel(x_label)
    ax.set_title(title);
    fig.savefig(img_name, dpi=fig.dpi)
    
    ws = wb.worksheets[0]
    img = openpyxl.drawing.image.Image(img_name)
    img.anchor = cell_num
    ws.add_image(img)
    wb.save(file_name)
    
def render_weekday(data, img_name, title, wb, cell_num, file_name):
    weekday_fig, weekday_ax = plt.subplots(figsize=(9,3))
    dataset_valuable=data.dropna(subset=['date_activity', 'date_person'])
    dataset_valuable['date_activity'] = pd.to_datetime(dataset_valuable['date_activity'])
    dataset_valuable['date_person'] = pd.to_datetime(dataset_valuable['date_person'])
    dataset_valuable=dataset_valuable.dropna(subset=['date_activity', 'date_person'])
    dataset_valuable.info()
    dataset_valuable['weekday'] = pd.to_datetime(dataset_valuable['date_person']).dt.day_name()
    sns.barplot(x=dataset_valuable['weekday'].value_counts(normalize=True).index.to_list(), y=dataset_valuable['weekday'].value_counts(normalize=True), ax=weekday_ax)
    weekday_ax.set_title(title)
    weekday_fig.savefig(img_name)
    ws = wb.worksheets[0]
    img = openpyxl.drawing.image.Image(img_name)
    img.anchor = cell_num
    ws.add_image(img)
    wb.save(file_name)
    
def render_activities_num_per_user(data, img_name, title, wb, cell_num, file_name):
    fig, ax = plt.subplots(figsize=(9,3))
    sns.distplot(data.groupby('people_id').count()['activity_id'], ax=ax, bins=range(0,35))
    ax.set_xlabel('number of activities per user')
    ax.set_xlim(1,35)
    _ = ax.set_xticks(range(1,35, 2))
    ax.set_title(title)
    fig.savefig(img_name, dpi=fig.dpi)
    ws = wb.worksheets[0]
    img = openpyxl.drawing.image.Image(img_name)
    img.anchor = cell_num
    ws.add_image(img)
    wb.save(file_name)
    
def render_activities_category_by_number(data, img_name, title, wb, cell_num, file_name):
    fig_act_category, ax_act_category = plt.subplots(figsize=(9,3))
    sns.barplot(x=data.activity_category.value_counts().index.to_list(), y=data.activity_category.value_counts(normalize=True), ax=ax_act_category)
    ax_act_category.set_title(title)
    fig_act_category.savefig(img_name, dpi=fig_act_category.dpi)
    ws = wb.worksheets[0]
    img = openpyxl.drawing.image.Image(img_name)
    img.anchor = cell_num
    ws.add_image(img)
    wb.save(file_name)
    
def render_activities_category_by_value_probability(data, img_name, title, wb, cell_num, file_name):
    fig_act_cat_value, ax_act_cat_value = plt.subplots(figsize=(9, 3))
    category_type_to_prediction_df = data.groupby('activity_category')['predictions'].mean()
    ax_act_cat_value = category_type_to_prediction_df.plot(kind='barh', figsize=(9,3), color=['r','b', 'y', 'k', 'grey'])
    ax_act_cat_value.set_ylabel('Probability to be valuable')
    ax_act_cat_value.set_xlabel('Activity category')
    ax_act_cat_value.set_title(title)
    _=plt.xticks([0.1,0.2,0.3,0.4,0.5, 0.6])
    fig_act_cat_value.savefig(img_name, dpi=fig_act_cat_value.dpi)
    ws = wb.worksheets[0]
    img = openpyxl.drawing.image.Image(img_name)
    img.anchor = cell_num
    ws.add_image(img)
    wb.save(file_name)

def display_logic(dataset, file_name):
    dataset['date_activity'] = pd.to_datetime(dataset['date_activity'])
    dataset['date_person'] = pd.to_datetime(dataset['date_person'])
    dataset=dataset.dropna()
    wb = openpyxl.Workbook()
    
    dataset_valuable = dataset.loc[dataset['predictions'] == 1]
    activities_per_day_v = dataset_valuable.groupby([pd.Grouper(key='date_activity', freq='1D')])['activity_id'].count().reset_index()
    render_datetime(activities_per_day_v, '# of activities', 'date', 'valuable_customers.png', 'A1',wb, 'Time-activities by valuable customers', file_name)
    
    dataset_unvaluable = dataset.loc[dataset['predictions'] == 0]
    activities_per_day_u = dataset_unvaluable.groupby([pd.Grouper(key='date_activity', freq='1D')])['activity_id'].count().reset_index()
    render_datetime(activities_per_day_u, '# of activities', 'date', 'unvaluable_customers.png', 'A15',wb, 'Time-activities by not valuable customers', file_name)
    
    render_weekday(dataset_valuable, 'weekday_valuable.png', 'Dayweek-activities by valuable customers', wb, 'A30',file_name)
    render_weekday(dataset_unvaluable, 'weekday_unvaluable.png', 'Dayweek-activities by unvaluable customers', wb, 'J30',file_name)


    render_activities_num_per_user(dataset_valuable, 'activities_per_valuable.png', 'User/activities-number by valuable customers', wb, 'A45',file_name)
    render_activities_num_per_user(dataset_unvaluable, 'activities_per_unvaluable.png', 'User/activities-number by unvaluable customers', wb, 'J45',file_name)

    render_activities_category_by_number(dataset_valuable, 'activities_category_per_valuable.png', 'Activity category by number of valuable users', wb, 'A60',file_name)
    render_activities_category_by_number(dataset_unvaluable, 'activities_category_per_unvaluable.png', 'Activity category by number of unvaluable users', wb, 'J60',file_name)

    render_activities_category_by_value_probability(dataset, 'activities_category_by_value.png', 'Activity category by probability to be valuable', wb, 'A75',file_name)
  
      
    fig_numeric_val, ax_numeric_val = plt.subplots(figsize=(24, 24))
    ax_numeric_val = dataset.groupby('char_38')['predictions'].mean().plot(kind='barh', figsize=(24, 24), color=['r','b', 'y', 'k', 'grey'])
    ax_numeric_val.set_title('bussiness value of char_38 column')
    ax_numeric_val.set_ylabel('fraction')
    _=plt.xticks([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])
    fig_numeric_val.savefig('char_38.png', dpi=fig_numeric_val.dpi)
    ws = wb.worksheets[0]
    img = openpyxl.drawing.image.Image('char_38.png')
    img.anchor = 'A90'
    ws.add_image(img)
    wb.save(file_name)

app.layout = html.Div(
    children=[
        html.H1(children="Client value analyst"),
        html.P(
            children=(
                "Analyze the value of clients for a company by"
                " their interaction with the app"
            ),
        ),
        dcc.Upload(
            id='upload-data',        
            children=html.Div([
            'Drag and Drop or ',
            html.A('Select Files')
            ]),
            style={
            'width': '80%',
            'height': '30px',
            'lineHeight': '30px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px'
            },
        ),
        html.Div(children=[html.Span('Model type '),dcc.Dropdown(
            id="model-chooser",
            options=[
                {"label": region, "value": region}
                for region in ["automatic", "decision tree", "logistic regression", "neural network"]
            ],
            value="decision tree",
            clearable=False,
            className="dropdown",
            style={'width': '70%'}
        )], style={'display':'flex', 'align-items':'center', 'column-gap': '8px'}),
        html.Div(
            id='params',
            style={
            'width':'500px',
            'display': 'flex',
            'flex-direction': 'column',
            'background-color': '#D3D3D3',
            'justify-content': 'center'
            }
        ),
        html.Div(id='automatic-placeholder'),
        html.Div(
            id='decision-placeholder',
            style={
            'width':'500px',
            'display': 'flex',
            'flex-direction': 'column',
            'background-color': '#D3D3D3'       
            }),
        html.Div(id='regression-placeholder'),
        html.Div(id='neural-placeholder'),
        html.Div(id='main'),
        html.Div(id='main2'),
        html.Div(id='main final'),
        html.Div(id='dislayed_trained_tree'),
        html.Button('Submit', id='start'),
        dcc.Store(id='model-store'),
        dcc.Store(id='param-store'),
        dcc.Store(id='app-store'),
        dcc.Store(id='trained-tree'),
        dcc.Store(id='trained-regr'),
        dcc.Store(id='trained-neural'),
    ],
    style={
    'width':'500px',
    'display': 'flex',
    'flex-direction': 'column',
    'background-color': '#D3D3D3',
    'column-gap': '10px',
    'row-gap': '15px',
    'border-radius': '15px'
    }
)

@callback(
    Output('model-store', 'data'),
    Output('params', 'children'),
    Input("model-chooser", "value"))
def on_model_change_callback(model_type):
      if(model_type == "automatic"):
          return model_type, html.Div('')    
      if(model_type == "decision tree"):
          return model_type, html.Div([
            html.Div(children=[
            html.Span('Tree max depth:'),
            dcc.Input(
            id="decision-max-depth",
            style={'height': '15px'}
            ),
            ],
            style={'display': 'flex', 'column-gap': '10px'}),
            html.Div(children=[
            html.Span('Tree max leaves number:'),
            dcc.Input(
            id="decision-max-leaves",
            style={'height': '15px'}
            ),
            ],
            style={'display': 'flex', 'column-gap': '10px'}), 
            html.Div(children=[
            html.Span('Sample of data for a leave:'),
            dcc.Input(
            id="decision-min-sample-leaf",
            style={'height': '15px'}
            ),
            ],
            style={'display': 'flex', 'column-gap': '10px'}), 
            dcc.Checklist(
            id="decision-optimizer",
            options=[
                {"label": "XGBoosting optimizer", "value": True}
            ],
            value=[False]
            ),
          ],
            style={
            'width':'500px',
            'display': 'flex',
            'flex-direction': 'column',
            'background-color': '#D3D3D3',
            'justify-content': 'center',
            'row-gap': '5px'
            })
      if(model_type == "logistic regression"):
          return model_type, html.Div([
            html.Div(children=[
            html.Span('Regularization type:'),
            dcc.Dropdown(
                id="logistic-regression-regularization",
                options=[
                    {"label": region, "value": region}
                    for region in ["l1", "l2"]
                ],
                value="l1",
                clearable=False,
                className="dropdown",
            )]),
            html.Div(children=[
            html.Span('Regularization coeff:'),
            dcc.Input(
            id="logistic-regression-regularization-coeff",
            style={'height': '15px'}
            ),
            ]),
          ],
          style={
          'width':'500px',
          'display': 'flex',
          'flex-direction': 'column',
          'background-color': '#D3D3D3',
          'justify-content': 'center',
          'row-gap': '5px'
          }
          )
      if(model_type == "neural network"):
          return model_type, html.Div([
              html.Div(children=[
              html.Span('Optimization algorithm:'),
              dcc.Dropdown(
                id="neural-optimization-alg",
                options=[
                    {"label": region, "value": region}
                    for region in ["Adam", "SGD"]
                ],
                value="SGD",
                clearable=False,
                className="dropdown",
              )]
              ),
              html.Div(children=[
              html.Span('Learning rate:'),
              dcc.Input(
              id="neural-lr"
              )
              ]),
              html.Div(children=[
              html.Span('Number of layers:'),
              dcc.Input(
              id="neural-max-depth"
              )]),
              html.Div(children=[
              html.Span('Number of neurons for the layer:'),
              dcc.Input(
              id="neural-logits-for-layer"
              )]),
              html.Div(children=[
              html.Span('Activation type:'),
              dcc.Dropdown(
              id="neural-activation-func",
              options=[
                {"label": region, "value": region}
                for region in ["relu", "tanh", "sigmoid"]
              ],
              value="relu",
              clearable=False,
              className="dropdown",
              )]),
          ],
          style={
            'width':'500px',
            'display': 'flex',
            'flex-direction': 'column',
            'background-color': '#D3D3D3',
            'justify-content': 'center',
            'row-gap': '10px'
          }
          )
          
def parse_contents(contents, filename, date):
    content_type, content_string = contents.split(',')
    try:
        if 'csv' in filename:
            # Assume that the user uploaded a CSV file
            df = pd.read_csv(filename).head(10000)
        elif 'xls' in filename:
            # Assume that the user uploaded an excel file
            df = pd.read_excel(io.BytesIO(decoded))
        return df
    except Exception as e:
        print(e)
        return html.Div([
            'There was an error processing this file.'
        ])

@callback(
    Output('automatic-placeholder', 'children'),
    Input('start', 'n_clicks'),
    Input('model-chooser', 'value'),
    State('upload-data', 'contents'),
    State('upload-data', 'filename'),
    State('upload-data', 'last_modified')
)
def automatic_params_aggregator(number_clicks, model_type, list_of_contents, list_of_names, list_of_dates):
    if(model_type != "automatic"):
        return html.Div('')
    df_people=people
    if list_of_contents is not None:
        df_people = parse_contents(list_of_contents, list_of_names, list_of_dates)
        print(df_people)
    try:
        accuracy_decision, f1_decision, precision_decision, recall_decision, predicted_dataset_decision = decision_logic()
        accuracy_regression, f1_regression, precision_regression, recall_regression, predicted_dataset_regression = logistic_regression_logic()
        accuracy_neural, f1_neural, precision_neural, recall_neural, predicted_dataset_neural = neural_network_logic()
        if(accuracy_decision > accuracy_regression and accuracy_decision > accuracy_neural):
                   best_score = 'Decision tree has the best scores'
                   dataset_final = predicted_dataset_decision
        elif(accuracy_regression > accuracy_decision and accuracy_regression > accuracy_neural):
                   best_score = 'Logistic regression has the best scores'
                   dataset_final = predicted_dataset_regression
        else:
                   best_score = 'Neural network has the best scores'
                   dataset_final = predicted_dataset_neural
        return html.Div(
        children=[
         html.Div(children=[
         html.Span(best_score
         ),
         html.Div(children=[   
         html.Span('Accuracy: '),   
         html.Span(max(accuracy_decision, accuracy_regression, accuracy_neural)),
         ],
         style={'background-color':'white', 'padding':'0 10px 10px 0'}
         ),
         html.Div(children=[ 
         html.Span('F1 score: '),
         html.Span(max(f1_decision, f1_regression, f1_neural))
         ],
         style={'background-color':'white'}
         ),
         html.Div(children=[ 
         html.Span('Precision: '),
         html.Span(max(precision_decision, precision_regression, precision_neural))
         ],
         style={'background-color':'white'}
         ),
         html.Div(children=[ 
         html.Span('Recall: '),
         html.Span(max(recall_decision, recall_regression, recall_neural))
         ],
         style={'background-color':'white'}
         )
         ],
         style={'display': 'flex', 'column-gap':'10px', 'flex-wrap':'wrap'}
         ),
        html.Div(
        [dash_table.DataTable(
            data=dataset_final.to_dict('rows'),
            columns=[{"name": i, "id": i,} for i in (dataset_final.columns)],
            id='3',
        )], style={'overflow':'scroll', 'width':'450px', 'height': '350px'}
        )  
        ],
        )
    except Exception as e:
        print(e)
        return html.Div(children=[
            html.Span('There was an error processing this file: ', style={'color':'red'}),
            html.Span(str(e))
        ],
        style={'background-color':'white', 'padding':'5px 10px 5px 10px'}               
        )

@callback(
    Output('decision-placeholder', 'children'),
    Input('start', 'n_clicks'),
    Input('model-chooser', 'value'),
    State('upload-data', 'contents'),
    State("decision-max-depth", "value"),
    State("decision-max-leaves", "value"),
    State("decision-min-sample-leaf", "value"),
    State("decision-optimizer", "value"),
    State('upload-data', 'filename'),
    State('upload-data', 'last_modified')
)

def decision_params_aggregator(number_clicks, model_type, list_of_contents, max_depth, max_leaves, min_sample, with_optimizer, list_of_names, list_of_dates):
    if(model_type != "decision tree"):
        return html.Div('')
    df_people=people
    if list_of_contents is not None:
        df_people = parse_contents(list_of_contents, list_of_names, list_of_dates)
    try: 
        accuracy, f1, precision, recall, predicted_dataset = decision_logic(int(max_depth) if max_depth != None else None, int(max_leaves) if max_leaves != None else None, bool(with_optimizer[-1]), int(min_sample) if min_sample != None else None, df_people)
        print('DDDD', predicted_dataset, 'decision_params_aggregator.xlsx')
        display_logic(predicted_dataset, 'decision_params_aggregator.xlsx')
        return html.Div(
        children=[
        html.Div(children=[  
         html.Div(children=[   
         html.Span('Accuracy: '),   
         html.Span(accuracy),
         ],
         style={'background-color':'white'}        
         ),
         html.Div(children=[ 
         html.Span('F1 score: '),
         html.Span(f1)
         ],
         style={'background-color':'white'}
         ),
         html.Div(children=[ 
         html.Span('Precision: '),
         html.Span(precision)
         ],
         style={'background-color':'white'}
         ),
         html.Div(children=[ 
         html.Span('Recall: '),
         html.Span(recall)
         ],
         style={'background-color':'white'}
         )],
        style={'display': 'flex', 'column-gap':'10px'}        
        ),
        html.Div(
        [dash_table.DataTable(
            data=predicted_dataset.to_dict('rows'),
            columns=[{"name": i, "id": i,} for i in (predicted_dataset.columns)],
            id='3',
        )], style={'overflow':'scroll', 'width':'450px', 'height': '350px'}
        )
        ],
        )
    except Exception as e:
        print(e)
        return html.Div(children=[
            html.Span('There was an error processing this file: ', style={'color':'red'}),
            html.Span(str(e))
        ],
        style={'background-color':'white', 'padding':'5px 10px 5px 10px'}               
        )
    
@callback(
    Output('regression-placeholder', 'children'),
    Input('start', 'n_clicks'),
    Input('model-chooser', 'value'),
    State('upload-data', 'contents'),
    State("logistic-regression-regularization", "value"),
    State("logistic-regression-regularization-coeff", "value"),
    State('upload-data', 'filename'),
    State('upload-data', 'last_modified')
)
def regression_params_aggregator(number_clicks, model_type, list_of_contents, regularization, regularization_coeff, list_of_names, list_of_dates):
    if(model_type != "logistic regression"):
        return html.Div('')
    if(regularization_coeff == None or  isinstance(regularization_coeff, str)):
        regularization_coeff = None
    df_people=people
    if list_of_contents is not None:
        df_people = parse_contents(list_of_contents, list_of_names, list_of_dates)
    try:
        accuracy, f1, precision, recall, predicted_dataset = logistic_regression_logic(regularization, float(regularization_coeff) if regularization_coeff != None and float(regularization_coeff) <= 1 and float(regularization_coeff)> 0 else 0.01, df_people)
        display_logic(predicted_dataset, 'regression_params_aggregator.xlsx')
        return html.Div(
        children=[
        html.Div(children=[
         html.Div(children=[   
         html.Span('Accuracy: '),   
         html.Span(accuracy),
         ],
         style={'background-color':'white'}        
         ),
         html.Div(children=[ 
         html.Span('F1 score: '),
         html.Span(f1)
         ],
         style={'background-color':'white'}
         ),
         html.Div(children=[ 
         html.Span('Precision: '),
         html.Span(precision)
         ],
         style={'background-color':'white'}
         ),
         html.Div(children=[ 
         html.Span('Recall: '),
         html.Span(recall)
         ],
         style={'background-color':'white'}
         )
        ],
        style={'display': 'flex', 'column-gap':'10px'}        
        ),
        html.Div(
        [dash_table.DataTable(
            data=predicted_dataset.to_dict('rows'),
            columns=[{"name": i, "id": i,} for i in (predicted_dataset.columns)],
            id='3',
        )], style={'overflow':'scroll', 'width':'450px', 'height': '350px'}
        )])
    except Exception as e:
        print(e)
        return html.Div(children=[
            html.Span('There was an error processing this file: ', style={'color':'red'}),
            html.Span(str(e))
        ],
        style={'background-color':'white', 'padding':'5px 10px 5px 10px'}               
        )
    
@callback(
    Output('neural-placeholder', 'children'),
    Input('start', 'n_clicks'),
    Input('model-chooser', 'value'),
    State('upload-data', 'contents'),
    State("neural-optimization-alg", "value"),
    State("neural-lr", "value"),
    State("neural-max-depth", "value"),
    State("neural-logits-for-layer", "value"),
    State("neural-activation-func", "value"),
    State('upload-data', 'filename'),
    State('upload-data', 'last_modified')
)
def neural_params_aggregator(number_clicks, model_type, list_of_contents, opt_alg, lr, max_depth, logits_for_layer, activation_func, list_of_names, list_of_dates):
    if(model_type != "neural network"):
        return html.Div('')
    df_people=people
    if list_of_contents is not None:
        df_people = parse_contents(list_of_contents, list_of_names, list_of_dates)
    try:
        accuracy, f1, precision, recall, predicted_dataset = neural_network_logic(opt_alg, float(lr) if lr != None else None, int(max_depth) if max_depth != None else None, int(logits_for_layer) if logits_for_layer != None else None, activation_func, df_people)
        display_logic(predicted_dataset, 'neural_params_aggregator.xlsx')
        return html.Div(
        children=[
        html.Div(children=[
         html.Div(children=[   
         html.Span('Accuracy: '),   
         html.Span(accuracy),
         ],
         style={'background-color':'white'}
         ),
         html.Div(children=[ 
         html.Span('F1 score: '),
         html.Span(f1)
         ],
         style={'background-color':'white'}
         ),
         html.Div(children=[ 
         html.Span('Precision: '),
         html.Span(precision)
         ],
         style={'background-color':'white'}
         ),
         html.Div(children=[ 
         html.Span('Recall: '),
         html.Span(recall)
         ],
         style={'background-color':'white'}
         )
        ], style={'display': 'flex', 'column-gap':'10px'}),
        html.Div(
        [dash_table.DataTable(
            data=predicted_dataset.to_dict('rows'),
            columns=[{"name": i, "id": i,} for i in (predicted_dataset.columns)],
            id='3',
        )], style={'overflow':'scroll', 'width':'450px', 'height': '350px'}
        )])
    except Exception as e:
        print(e)
        return html.Div(children=[
            html.Span('There was an error processing this file: ', style={'color':'red'}),
            html.Span(str(e))
        ],
        style={'background-color':'white', 'padding':'5px 10px 5px 10px'}               
        )
    
if __name__ == "__main__":
    app.run_server(debug=True,dev_tools_ui=False)