import streamlit as st
from analytical_sol import analytical_plot, trajectory_plot
from ann_sol import ann_plot, predictions
import json
import numpy as np

st.set_page_config(page_title='Sea Turtle Prediction',
                   page_icon=':chart_with_upwards_trend:',
                   layout='wide',
                   initial_sidebar_state='expanded')

####################### Sidebar ####################################
st.sidebar.markdown('<style>div.block-container{padding-top:25px;}</style>', unsafe_allow_html=True)
# domain setting
st.sidebar.title('Sea Turtle Prediction `version 1.0`')
st.sidebar.header('Predator-Prey Domain Setting')
init_col1, init_col2, init_col3 = st.sidebar.columns(3)
x0 = init_col1.text_input('Initial eggs', 1000)
y0 = init_col2.text_input('Initial jouvenils', 100)
z0 = init_col3.text_input('Initial predator', 10)

t_max = st.sidebar.text_input('Enter max `t` value', 50)
n_t = st.sidebar.text_input('Enter total data size', 1000)

# parameters setting
st.sidebar.header('Predator-Prey Parameters setting')

# egg parameters
st.sidebar.subheader('Egg parameters')

K_value = st.sidebar.text_input('Enter Carrying capacity (`K`) value', 700)
egg_scol1, egg_scol2, egg_scol3 = st.sidebar.columns(3)
a_value = egg_scol1.slider('Egg failing rate (`a`) value', min_value=0.0, max_value=1.0, step=0.001, value=0.2)
r_value = egg_scol2.slider('Eggs laid rate (`r`) value', min_value=0.0, max_value=2.0, step=0.001, value=1.3)
w_value = egg_scol3.slider('Successful rate (`w`) value', min_value=0.0, max_value=1.0, step=0.001, value=0.53)

# jouvenile parameters
st.sidebar.subheader('Jouvenile parameters')
jou_scol1, jou_scol2 = st.sidebar.columns(2)
c_value = jou_scol1.slider('Survival rate (`c`) value', min_value=0.0, max_value=1.0, step=0.001, value=0.001)
g_value = jou_scol2.slider('Conversion rate (`g`) value', min_value=0.0, max_value=1.0, step=0.001, value=1.0)

# predator parameters
st.sidebar.subheader('Predator parameters')
pre_scol1, pre_scol2, pre_scol3 = st.sidebar.columns(3)
b_value = pre_scol1.slider('Predation rate (`b`) value', min_value=0.0, max_value=1.0, step=0.001, value=0.15)
s_value = pre_scol2.slider('Natural death rate (`s`) value', min_value=0.0, max_value=1.0, step=0.001, value=0.7)
f_value = pre_scol3.slider('Energy conversion rate (`f`) value', min_value=0.0, max_value=2.0, step=0.001, value=1.0)


################################ Main dashboard ################################

st.title('🐢 Sea Turtle Predator-Prey Prediction')
st.markdown('<style>div.block-container{padding-top:20px;}</style>', unsafe_allow_html=True)

st.markdown('### SODE Domain Metrics')
st.markdown('#### Initial values')
init_main_col1, init_main_col2, init_main_col3 = st.columns(3)
init_main_col1.metric('Initial eggs', x0)
init_main_col2.metric('Initial jouvenils', y0)
init_main_col3.metric('Initial predator', z0)

st.markdown('#### Domain values')
dom_col1, dom_col2 = st.columns(2)
dom_col1.metric('Maximum time (`t`) value', t_max)
dom_col2.metric('Domain data size', n_t)

st.markdown('### SODE Parameters Metrics')
st.markdown('#### Eggs parameters')
egg_col1, egg_col2, egg_col3, egg_col4 = st.columns(4)
egg_col1.metric('Eggs laid rate', r_value)
egg_col2.metric('Carrying capacity', K_value)
egg_col3.metric('Failing hatch rate', a_value)
egg_col4.metric('Succesful hatchling rate', w_value)

st.markdown('#### Jouveniles parameters')
jov_col1, jov_col2 = st.columns(2)
jov_col1.metric('Conversion rate', g_value)
jov_col2.metric('Survival rate', c_value)

st.markdown('#### Predators parameters')
pre_col1, pre_col2, pre_col3 = st.columns(3)
pre_col1.metric('Energy conversion rate', f_value)
pre_col2.metric('Predation rate', b_value)
pre_col3.metric('Predator natural death rate', s_value)

parameters_value = {
    "a_value": a_value,
    "b_value": b_value,
    "c_value": c_value,
    "f_value": f_value,
    "g_value": g_value,
    "K_value": int(K_value),
    "r_value": r_value,
    "s_value": s_value,
    "w_value": w_value,
    "T": t_max,
    "n_t": n_t,
    "x0": x0,
    "y0": y0,
    "z0": z0
}

file_path = "parameters_value.json"

with open(file_path, "w") as json_file:
    json.dump(parameters_value, json_file, indent=4)

st.markdown('### Analytical solution plot')

analytical_plot(int(x0), int(y0), int(z0), int(n_t), int(t_max))

tab1, tab2, tab3, tab4 = st.tabs(["Run Model", "Build Model", "Prediction", "Initial Value Simulation"])

tab1.subheader("Run a pre-train model")
################ tab 1 ############################

init_main_col1, init_main_col2, init_main_col3, init_main_col4 = tab1.columns(4)
init_main_col1.metric('Hidden Layers', 10)
init_main_col2.metric('Neurons', 150)
init_main_col3.metric('Optimizer', 'Adam')
init_main_col4.metric('Apply Restarting', 'Yes')

ann_cycles = tab1.text_input('Enter the number of cycle', 5,  key="old_cycle")
ann_iters = tab1.text_input('Enter the maximum number of iteration', 100)

_, _, _, train_btn_mid, _, _, _ = tab1.columns(7)
train_btn = train_btn_mid.button('Train Model')

plot_ann = st.empty()

if train_btn:
    ann_architecture = {
    "layers": 10,
    "neurons": 150,
    "iterations": ann_iters,
    "apply_restarting": 'Yes',
    "cycles": ann_cycles,
    "optimizer": 'Adam',
    'new_model': 'No'
    }
    file_path = "ann_architecture.json"
    with open(file_path, "w") as json_file:
        json.dump(ann_architecture, json_file, indent=4)
    ann_plot(int(x0), int(y0), int(z0), int(n_t), int(t_max), tab1)



################ tab 2 ############################

tab2.subheader("Create your own model")

ann_col1, ann_col2, ann_col3, ann_col4 = tab2.columns(4)
ann_layers = ann_col1.text_input('Enter the number of hidden layer', 1)
ann_neurons = ann_col2.text_input('Enter the number of neurons each layer', 10)
ann_iters = ann_col3.text_input('Enter the maximum number of iteration', 10)
ann_cycles = 1
apply_restarting = ann_col4.selectbox(
    'Apply Restarting?',
    ('Please Choose',
     'Yes',
    'No'))

if apply_restarting=='Yes':
    ann_cycles = tab2.text_input('Enter the number of cycle', 5)
else:
    tab2.empty()

ann_optimizer = tab2.selectbox(
    'Select the optimizer',
    ('Adadelta',
    'Adagrad',
    'Adam',
    'Adamax',
    'Ftrl',
    'Nadam',
    'RMSprop',
    'SGD'))

_, _, _, build_btn_mid, _, _, _ = tab2.columns(7)
build_btn = build_btn_mid.button('Build New Model')

if build_btn:
    new_model = 'Yes'
    ann_architecture = {
    "layers": ann_layers,
    "neurons": ann_neurons,
    "iterations": ann_iters,
    "apply_restarting": apply_restarting,
    "cycles": ann_cycles,
    "optimizer": ann_optimizer,
    'new_model': 'Yes'
    }
    file_path = "ann_architecture.json"
    with open(file_path, "w") as json_file:
        json.dump(ann_architecture, json_file, indent=4)
    ann_plot(int(x0), int(y0), int(z0), int(n_t), int(t_max), tab2)


################ tab 3 ############################
tab3.markdown('### Model Prediction')

time_value = tab3.slider('Enter time value', min_value=0, max_value=int(t_max), step=1, value=0)

_, _, _, pred_btn_mid, _, _, _ = tab3.columns(7)
pred_btn = pred_btn_mid.button('Predict')



if pred_btn:
    results= predictions(time_value)
    tab3.markdown('#### Prediction result')
    result_col1, result_col2, result_col3 = tab3.columns(3)
    result_col1.metric('Eggs', "{:.2f}".format(results[0][0]))
    result_col2.metric('Jouvenils', "{:.2f}".format(results[0][1]))
    result_col3.metric('Predator', "{:.2f}".format(results[0][2]))

################ tab 4 ############################
tra_col1, tra_col2, tra_col3 = tab4.columns(3)

# trajectory plot parameters
tab4.markdown('### Initial value simulation')
tab4.markdown('#### Eggs')
x0_col1, x0_col2, x0_col3 = tab4.columns(3)
x0_min = x0_col1.text_input('Mininum value', 0, key="x0_min")
x0_max = x0_col2.text_input('Maximum value', x0, key="x0_max")
x0_n = x0_col3.text_input('How many?', 10, key="x0_n")

tab4.markdown('#### Jouveniles')
y0_col1, y0_col2, y0_col3 = tab4.columns(3)
y0_min = y0_col1.text_input('Mininum value', 0, key="y0_min")
y0_max = y0_col2.text_input('Maximum value', y0, key="y0_max")
y0_n = y0_col3.text_input('How many?', 10, key="y0_n")

tab4.markdown('#### Predators')
z0_col1, z0_col2, z0_col3 = tab4.columns(3)
z0_min = z0_col1.text_input('Mininum value', 0, key="z0_min")
z0_max = z0_col2.text_input('Maximum value', z0, key="z0_max")
z0_n = z0_col3.text_input('How many?', 10, key="z0_n")

plot_iter = tab4.text_input('Input maximum iteration to plot', 10)

_, _, _, traj_btn_mid, _, _, _ = tab4.columns(7)
traj_btn = traj_btn_mid.button('Plot')

if traj_btn:
    trajectory_plot(int(x0), int(y0), int(z0), int(n_t), int(t_max),
                int(x0_min), int(x0_max), int(x0_n), 
                int(y0_min), int(y0_max), int(y0_n),
                int(z0_min), int(z0_max), int(z0_n), int(plot_iter), tab4)


    
