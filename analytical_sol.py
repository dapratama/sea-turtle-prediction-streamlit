import numpy as np
from scipy.integrate import odeint
import plotly.graph_objects as go
import streamlit as st
import pandas as pd
import json
import time
import tensorflow as tf
from keras.models import load_model
from tensorflow import keras
from get_function import build_model, get_model, get_architecture, get_value



# define the lotka-voltera system

def fx(init, data):

    a, b, c, f, g, K, r, s, w, _, _, _, _, _ = get_value()

    dxdt = r*init[0]*(1-(init[0]/K))-a*init[0]-w*init[0] 
    dydt = g*w*init[0]-c*init[1]-b*init[1]*init[2]
    dzdt = f*b*init[1]*init[2]-s*init[2]
    fx = np.array([dxdt, dydt, dzdt])
    return fx

def analytical_plot(x0, y0, z0, nt, T):
 
    data = np.linspace(0, T, nt)

    # define the parameters

    init = np.array([x0, y0, z0])
    # exact solution
    exact_gt = odeint(fx, init, data)

    t_data = pd.DataFrame(data)
    x_data = pd.DataFrame(exact_gt[:, :1])
    y_data = pd.DataFrame(exact_gt[:, 1:2])
    z_data = pd.DataFrame(exact_gt[:, 2:3])

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t_data[0], y=x_data[0],
                        mode='lines', name='Eggs analytic solutions'))
    fig.add_trace(go.Scatter(x=t_data[0], y=y_data[0],
                        mode='lines', name='Jouveniles analytic solutions'))
    fig.add_trace(go.Scatter(x=t_data[0], y=z_data[0],
                        mode='lines', name='Predators analytic solutions'))
    
    fig.update_layout(
        title={'text':"Sea turtles predator-prey analytical solution", 'y':0.9, 'x':0.25},
        autosize=False,
        width=800,
        height=500,
        yaxis=dict(
            title_text="T (time)",
            titlefont=dict(size=20),
        ),
        xaxis=dict(
            title_text="Value",
            titlefont=dict(size=20),
        )
    )

    return st.plotly_chart(fig, use_container_width=True)

def trajectory_plot(x0, y0, z0, nt, T,
                    x0_min, x0_max, x0_n,
                    y0_min, y0_max, y0_n,
                    z0_min, z0_max, z0_n, iter, tab):
    
    _, _, _, _, _, _, new_model = get_architecture()

    if new_model=='Yes':
        model = get_model(new_model)
    else:
        model = get_model(new_model)
 
    data = np.linspace(0, T, nt)

    x0_sim = np.linspace(x0_min, x0_max, x0_n)
    y0_sim = np.linspace(y0_min, y0_max, y0_n)
    z0_sim = np.linspace(z0_min, z0_max, z0_n)

    figx = go.Figure()
    figy = go.Figure()
    figz = go.Figure()

    for i in x0_sim:
        with st.spinner('Plotting simulation...'):
            initx = np.array([i, y0, z0])
            # exact solution
            exact_gtx = odeint(fx, initx, data)
            model.fit(data, exact_gtx, epochs=iter, verbose=0)
            predictions_x = model.predict(data)
        
            t_data = pd.DataFrame(data)
            x_data = pd.DataFrame(predictions_x[:, :1])

            
            figx.add_trace(go.Scatter(x=t_data[0], y=x_data[0],
                                mode='lines', name='Initial value {}'.format(int(i))))
        
    for j in y0_sim:
        with st.spinner('Plotting simulation...'):
            inity = np.array([x0, j, z0])
            # exact solution
            exact_gty = odeint(fx, inity, data)
            model.fit(data, exact_gty, epochs=iter, verbose=0)
            predictions_y = model.predict(data)

            t_data = pd.DataFrame(data)
            y_data = pd.DataFrame(predictions_y[:, 1:2])

            
            figy.add_trace(go.Scatter(x=t_data[0], y=y_data[0],
                                mode='lines', name='Initial value {}'.format(int(j))))
        
    for k in z0_sim:
        with st.spinner('Plotting simulation...'):
            initz = np.array([x0, y0, k])
            # exact solution
            exact_gtz = odeint(fx, initz, data)
            model.fit(data, exact_gtz, epochs=iter, verbose=0)
            predictions_z = model.predict(data)

            t_data = pd.DataFrame(data)
            z_data = pd.DataFrame(predictions_z[:, 2:3])

            
            figz.add_trace(go.Scatter(x=t_data[0], y=z_data[0],
                                mode='lines', name='Initial value {}'.format(int(k))))
    
    figx.update_layout(
        title={'text':"Eggs Initial Value Simulation", 'y':0.9, 'x':0.25},
        autosize=False,
        width=800,
        height=500,
        yaxis=dict(
            title_text="T (time)",
            titlefont=dict(size=20),
        ),
        xaxis=dict(
            title_text="Value",
            titlefont=dict(size=20),
        )
    )

    figy.update_layout(
        title={'text':"Jouveniles Initial Value Simulation", 'y':0.9, 'x':0.25},
        autosize=False,
        width=800,
        height=500,
        yaxis=dict(
            title_text="T (time)",
            titlefont=dict(size=20),
        ),
        xaxis=dict(
            title_text="Value",
            titlefont=dict(size=20),
        )
    )

    figz.update_layout(
        title={'text':"Predators Initial Value Simulation", 'y':0.9, 'x':0.25},
        autosize=False,
        width=800,
        height=500,
        yaxis=dict(
            title_text="T (time)",
            titlefont=dict(size=20),
        ),
        xaxis=dict(
            title_text="Value",
            titlefont=dict(size=20),
        )
    )

    tra_col1, tra_col2, tra_col3 = tab.columns(3)
    with tra_col1:
        st.plotly_chart(figx, use_container_width=True)

    with tra_col2:
        st.plotly_chart(figy, use_container_width=True)
    
    with tra_col3:
        st.plotly_chart(figz, use_container_width=True)


# Define a custom loss function
def mse_loss(u_true, u_pred):
    return tf.reduce_mean(tf.square(u_true[0] - u_pred[0])) + tf.reduce_mean(tf.square(u_true[1:] - u_pred[1:]))

keras.utils.get_custom_objects()["mse_loss"] = mse_loss