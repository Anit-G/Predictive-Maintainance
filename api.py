from Helper import *
from Helper import r2_keras
import time
from flask import Flask, request, render_template
from tensorflow.keras.models import load_model

app = Flask(__name__)
data_link = './Data/'
train_link = 'train_FD001.txt'
rul_link = "RUL_FD001.txt"


def data_load(state):
    if state==1:
        test_link = 'test_FD001.txt'
        constant_columns = ['sensor3','sensor4','sensor8','sensor13','sensor22','sensor19']
        lowcol_columns = ['sensor1','sensor2','sensor9','sensor17']
    elif state==2:
        test_link = 'test_FD002.txt'
        constant_columns = ['sensor3','sensor22','sensor19']
        lowcol_columns = ['sensor1','sensor2','sensor4','sensor5','sensor8','sensor9','sensor10','sensor11','sensor13','sensor15','sensor21']
    elif state==3:
        test_link = 'test_FD003.txt'
        constant_columns = ['sensor3','sensor4','sensor8','sensor13','sensor22','sensor19']
        lowcol_columns = ['sensor1','sensor2','sensor9','sensor18','sensor23','sensor24']
    elif state ==4:
        test_link = 'test_FD004.txt'
        constant_columns = ['sensor3','sensor22','sensor19']
        lowcol_columns = ['sensor4','sensor8','sensor9','sensor10','sensor15','sensor16']
    else:
        print('Invalid')
        logger.warning('This is an Invalid State must be 1,2,3 or 4')

    logger.info('Start Importing Data') 
    fd1t = pd.read_csv(data_link+test_link,sep=" ",header=None)
    # Dropping columns 26 and 27 cuz they are NaN
    logger.info('Drop NaN columns')
    fd1t.drop(columns=[26,27],inplace=True)

    columns = ['unit_number','time_in_cycles','sensor1','sensor2','sensor3','sensor4','sensor5','sensor6','sensor7','sensor8','sensor9','sensor10','sensor11',
    'sensor12','sensor13','sensor14','sensor15','sensor16','sensor17','sensor18','sensor19','sensor20','sensor21','sensor22','sensor23','sensor24' ]
    fd1t.columns = columns
    # Delete columns with constant values
    logger.info('Delete columns with constant values') 
    fd1t.drop(columns=constant_columns,inplace=True)
    # Delete Columns with low correlation ~0
    logger.info('Delete columns with low correlation')
    fd1t = fd1t.drop(columns = lowcol_columns)

    # RUL
    logger.info('Import RUL') 
    RUL = pd.read_csv(data_link+rul_link,sep=" ",header=None)

    #Prepare blank train set for preprocessing step:
    logger.info('Creating Blank Set')
    fd1 = pd.read_csv(data_link+train_link,sep=" ",header=None)
    fd1.drop(columns=[26,27],inplace=True)
    fd1.columns = columns
    fd1.drop(columns=constant_columns,inplace=True)
    fd1 = prepare_train_data(fd1)
    train_df = fd1.drop(columns = lowcol_columns)

    seq_array, label_array, lstm_test_df, sequence_length, sequence_cols = lstm_data_preprocessing(train_df, fd1t, RUL.copy())
    return lstm_test_df, sequence_length,sequence_cols

def load_mdl(state):
    if state==1:
        model_name = './model/model1.h5'
    elif state==2:
        model_name = './model/model2.h5'
    elif state==3:
        model_name = './model/model3.h5'
    elif state==4:
        model_name = './model/model4.h5'
    else:
        pass
    # model variable refers to the global variable
    # load model
    dependinces = {'r2_keras': r2_keras}
    model = load_model(model_name,custom_objects=dependinces)
    return model

def html_table(pred):
  print('<table>')
  for n,p in enumerate(pred):
    print('  <tr><td>')
    print('    </td><td>'.join([n,p]))
    print('  </td></tr>')
  print('</table>')

@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def predict():
    # Works only for a single sample
    state = [float(x) for x in request.form.values()]
    if request.method == 'POST':
        model = load_mdl(state[0])  # load model at the beginning once only
        lstm_test_df,sequence_length, sequence_cols = data_load(state[0])
        logger.info('##### LSTM Model Validation #####') 
        MAE, R2, pred = lstm_valid_evaluation(lstm_test_df, model, sequence_length, sequence_cols)
        pred = list(np.concatenate(pred.round()).astype(int))
        
    return render_template('result.html', prediction_text="Predicted RUL: {}".format(pred), mae="MAE: {}".format(MAE),r2="R2: {}".format(R2))



if __name__ == '__main__':
    
    # app.run(debug=True)
    app.run(host='localhost', port=5000)

#     <!-- <table>
#   {%for id, name in enumerate(prediction_text)%}
#   <tr>
#     <td> {{id}} </td>
#     <td> {{name}} </td>
#   </tr>
#     {%endfor%}
# </table> -->