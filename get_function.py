import json
import time
import streamlit as st
from keras.models import load_model, Sequential
from keras.layers import Dense


def get_model(new_model):
    global model
    if new_model=='Yes':
        with st.spinner('Loading model...'):
            time.sleep(2.5)
            model = load_model('new_model.h5')
    else:
        with st.spinner('Loading model...'):
            time.sleep(2.5)
            model = load_model('old_model.h5')
    return model

def get_architecture():
    with open('ann_architecture.json', "r") as json_file:
        architecture = json.load(json_file)
    ann_layers = architecture["layers"]
    ann_neurons = architecture["neurons"]
    ann_iters = architecture["iterations"]
    apply_restarting = architecture['apply_restarting']
    ann_cycles = architecture["cycles"]
    ann_optimizer = architecture["optimizer"]
    new_model = architecture['new_model']
    return ann_layers, ann_neurons, ann_iters, apply_restarting, ann_cycles, ann_optimizer, new_model

def get_value():
    with open('parameters_value.json', "r") as json_file:
        parameters = json.load(json_file)
    a_value = parameters["a_value"]
    b_value = parameters["b_value"]
    c_value = parameters["c_value"]
    f_value = parameters["f_value"]
    g_value = parameters["g_value"]
    K_value = parameters["K_value"]
    r_value = parameters["r_value"]
    s_value = parameters["s_value"]
    w_value = parameters["w_value"]
    T = parameters["T"]
    n_t = parameters['n_t']
    x0 = parameters['x0']
    y0 = parameters['y0']
    z0 = parameters['z0']
    return a_value, b_value, c_value, f_value, g_value, K_value, r_value, s_value, w_value, T, n_t, x0, y0, z0

def build_model(num_layers, neurons_per_layer):
    #initializers = keras.initializers.RandomUniform(minval=0., maxval=100.)
    with st.spinner('Creating model...'):
        time.sleep(2.5)
        model = Sequential()

        # Add the input layer
        model.add(Dense(neurons_per_layer, activation='relu', input_shape=(1,)))

        # Add hidden layers
        for _ in range(num_layers): 
            model.add(Dense(neurons_per_layer, activation='relu'))

        # Add the output layer
        model.add(Dense(3))  # Adjust the activation function as needed


    return model