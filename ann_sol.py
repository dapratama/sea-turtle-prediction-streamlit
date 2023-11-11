import time
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.integrate import odeint
from tensorflow import keras
from analytical_sol import fx, get_value

from tensorflow import keras
from get_function import get_value, get_architecture, get_model, build_model


class StreamlitProgressBarCallback(keras.callbacks.Callback):
    def __init__(self, max_epochs, current_cycle, max_cycles):
        super(StreamlitProgressBarCallback, self).__init__()
        _, _, _, _, _, _, _, _, _, self.T, self.nt, self.x0, self.y0, self.z0 = get_value()
        _, _, _, _, self.ann_cycles, _, _ = get_architecture()
        self.data = np.linspace(0, int(self.T), int(self.nt))
        self.max_epochs = max_epochs
        self.max_cycles = max_cycles
        self.current_cycle = current_cycle
        self.start_time = None
        self.loss_values = []
        self.mae_values = []
        self.prediction_plots = []
        self.training_completed = False
        self.progress_bar = None
        self.info_text = None
        self.info_cycle = None

    def on_train_begin(self, logs=None):
        self.progress_bar = st.progress(0)
        self.info_text = st.text('Training process begin. Please wait..')
        self.epochs_completed = 0
        self.loss_figure = st.empty()
        self.result_figure = st.empty()

    def on_epoch_end(self, epoch, logs=None):


        self.epochs_completed += 1
        progress_percent = self.epochs_completed / self.max_epochs
        self.progress_bar.progress(progress_percent)

        if logs is not None:
            loss = logs.get('loss')
            self.loss_values.append(loss)
            info = f'Iteration: {self.epochs_completed}/{self.max_epochs} - Completion: {int(progress_percent * 100)}% - Loss: {loss}'
            self.info_text.text(info)

            if self.training_completed:
                # Clear the figures and return to avoid plotting
                self.loss_figure.empty()
                self.result_figure.empty()
                self.info_text.empty()
                return

            self.plot_loss_graph()

    def on_train_end(self, logs=None):
        self.training_completed = True
        # Clear the loss figure
        self.loss_figure.empty()
        self.progress_bar.empty()
        self.info_text.empty()

    def plot_loss_graph(self):
        t_data = pd.DataFrame(self.data)
        init = np.array([int(self.x0), int(self.y0), int(self.z0)])
        exact_gt = odeint(fx, init, self.data)

        x_data = pd.DataFrame(exact_gt[:, :1])
        y_data = pd.DataFrame(exact_gt[:, 1:2])
        z_data = pd.DataFrame(exact_gt[:, 2:3])

        if self.loss_values:

            # loss plot figure
            figure = make_subplots(rows=1, cols=2, subplot_titles=("Loss Function Over Epochs", "Approximation Results"))
            epochs = list(range(1, len(self.loss_values) + 1))
            figure.append_trace(go.Scatter(x=epochs, y=self.loss_values, mode='lines', name='MSE', legendgroup='1'), row=1, col=1)
            figure.update_xaxes(title_text="Iterations", row=1, col=1)
            figure.update_yaxes(title_text="MSE", row=1, col=1)

            # prediction plot figure

            if self.model is not None:
                # Generate predictions
                predictions = self.model.predict(self.data)
                self.prediction_plots.append(predictions)

                if len(self.prediction_plots) > 1:
                    # Plot the latest predictions
                    last_predictions = self.prediction_plots[-1]

                    figure.append_trace(go.Scatter(x=t_data[0], y=x_data[0], mode='lines', name='act_x', legendgroup='2'), row=1, col=2)
                    figure.append_trace(go.Scatter(x=t_data[0], y=y_data[0], mode='lines', name='act_y', legendgroup='2'), row=1, col=2)
                    figure.append_trace(go.Scatter(x=t_data[0], y=z_data[0], mode='lines', name='act_z', legendgroup='2'), row=1, col=2)

                    figure.append_trace(go.Scatter(x=t_data[0], y=last_predictions[:, 0], mode='lines', line=dict(dash='dash'), name='x_predicted', legendgroup='2'), row=1, col=2)
                    figure.append_trace(go.Scatter(x=t_data[0], y=last_predictions[:, 1], mode='lines', line=dict(dash='dash'), name='y_predicted', legendgroup='2'), row=1, col=2)
                    figure.append_trace(go.Scatter(x=t_data[0], y=last_predictions[:, 2], mode='lines', line=dict(dash='dash'), name='z_predicted', legendgroup='2'), row=1, col=2)
                    figure.update_xaxes(title_text="Time", row=1, col=2)
                    figure.update_yaxes(title_text="Value", row=1, col=2)
                    figure.update_layout(legend_tracegroupgap = 50)
                    self.loss_figure.plotly_chart(figure, use_container_width=True)

def plot_final_results(t_data, loss_values, num_cycles, exact_gt, final_predictions, tab):
        t_data = pd.DataFrame(t_data)
        x_data = pd.DataFrame(exact_gt[:, :1])
        y_data = pd.DataFrame(exact_gt[:, 1:2])
        z_data = pd.DataFrame(exact_gt[:, 2:3])

        # loss plot figure

        figure = make_subplots(rows=1, cols=2, subplot_titles=("Loss Function Over Epochs", "Approximation Results"))

        x_values = []
        sum = 0
        for i in range(num_cycles):
            min_x = sum
            sum += len(loss_values[i])
            max_x = sum - 1
            x = np.linspace(min_x, max_x, max_x - min_x)
            y = np.random.uniform(1, 10, len(x))
            x_values.append(x)

        for i in range(num_cycles):
            figure.append_trace(go.Scatter(x=x_values[i], y=loss_values[i], mode='lines', name=f'Cycle {i + 1}', legendgroup = '1'), row=1, col=1)

        #epochs = list(range(1, len(loss_values) + 1))
        #figure.append_trace(go.Scatter(x=epochs, y=loss_values, mode='lines', name='MSE'), row=1, col=1)
        figure.update_xaxes(title_text="Iterations", row=1, col=1)
        figure.update_yaxes(title_text="MSE", row=1, col=1)

        figure.append_trace(go.Scatter(x=t_data[0], y=x_data[0], mode='lines', name='act_x', legendgroup = '2'), row=1, col=2)
        figure.append_trace(go.Scatter(x=t_data[0], y=y_data[0], mode='lines', name='act_y', legendgroup = '2'), row=1, col=2)
        figure.append_trace(go.Scatter(x=t_data[0], y=z_data[0], mode='lines', name='act_z', legendgroup = '2'), row=1, col=2)

        figure.append_trace(go.Scatter(x=t_data[0], y=final_predictions[:, 0], mode='lines', line=dict(dash='dash'), name='x_predicted', legendgroup = '2'), row=1, col=2)
        figure.append_trace(go.Scatter(x=t_data[0], y=final_predictions[:, 1], mode='lines', line=dict(dash='dash'), name='y_predicted', legendgroup = '2'), row=1, col=2)
        figure.append_trace(go.Scatter(x=t_data[0], y=final_predictions[:, 2], mode='lines', line=dict(dash='dash'), name='z_predicted', legendgroup = '2'), row=1, col=2)
        figure.update_xaxes(title_text="Time", row=1, col=1)
        figure.update_yaxes(title_text="Value", row=1, col=1)

        figure.update_layout(legend_tracegroupgap = 50)
        return tab.plotly_chart(figure, use_container_width=True)


def ann_plot(x0, y0, z0, nt, T, tab):

    data = np.linspace(0, T, nt)

    # define the parameters
    init = np.array([x0, y0, z0])

    # exact solution
    exact_gt = odeint(fx, init, data)

    # Create a neural network model
    ann_layers, ann_neurons, ann_iters, apply_restarting, ann_cycles, ann_optimizer, new_model = get_architecture()

    if new_model=='Yes':
        model = build_model(int(ann_layers), int(ann_neurons))
    else:
        model = get_model(new_model)

    # Compile the model
    earlystopping = keras.callbacks.EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
    model.compile(optimizer=ann_optimizer, loss=mse_loss)

    ann_iters = int(ann_iters)
    ann_cycles = int(ann_cycles)
    
    start_time = time.time()

    history_mse_loss = []
    history_mse_cycle = []
    

    if apply_restarting=='Yes':
        progress_cycle = tab.progress(0)
        # Train the model
        for i in range(ann_cycles): 
            
            progress_percent = i / ann_cycles
            progress_cycle.progress(progress_percent)
            if (history_mse_loss==[]):
                info = f'Cycles: {i} - Completion: {int(progress_percent * 100)}% - Current best loss: None'
            else:
                info = f'Cycles: {i} - Completion: {int(progress_percent * 100)}% - Current best loss: {min(history_mse_cycle)}'
            info_text = tab.text(info)

            history = model.fit(data, exact_gt, epochs=ann_iters, callbacks=[StreamlitProgressBarCallback(ann_iters, i, ann_cycles), earlystopping], verbose=0)
            history_mse_cycle.extend(history.history['loss'])
            history_mse_loss.append(history.history['loss'])
            
            info_text.empty()
        progress_percent = ann_cycles / ann_cycles
        progress_cycle.progress(progress_percent)
        info = f'Cycles: {ann_cycles} - Completion: {int(progress_percent * 100)}% - Best loss: {min(history_mse_loss[i])}'
        info_text = tab.text(info)
        
    else:
        for i in range(ann_cycles):
            history = model.fit(data, exact_gt, epochs=ann_iters, callbacks=[StreamlitProgressBarCallback(ann_iters, i, ann_cycles)], verbose=0)
            history_mse_loss.append(history.history['loss'])
        tab.progress(100)
        info = f'Iteration: {ann_iters} - Completion: {100}% - Loss: {min(history_mse_loss[i])}'
        info_text = tab.text(info)
    
    predictions = model.predict(data)
    end_time = time.time()
    elapsed_time = end_time - start_time
    minutes, seconds = divmod(int(elapsed_time), 60)
    tab.write(f"Training complete! Total elapsed time: {minutes} minutes {seconds} seconds")
    if new_model=='Yes':
        model.save('new_model.h5')
    else:
        model.save('old_model.h5')
    return plot_final_results(data, history_mse_loss, ann_cycles, exact_gt, predictions, tab)

def predictions(time):

    _, _, _, _, _, _, new_model = get_architecture()
    time = np.array(time).reshape(-1, 1)

    if new_model=='Yes':
        model = get_model(new_model)
        pred = model.predict(time)
    else:
        model = get_model(new_model)
        pred = model.predict(time)

    return pred

# Define a custom loss function
def mse_loss(u_true, u_pred):

    return tf.reduce_mean(tf.square(u_true[0] - u_pred[0])) + tf.reduce_mean(tf.square(u_true[1:] - u_pred[1:]))

keras.utils.get_custom_objects()["mse_loss"] = mse_loss