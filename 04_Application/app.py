import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import pydeck as pdk

import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots


# SETTING PAGE CONFIG TO WIDE MODE
st.set_page_config(layout="wide")


st.title('Get Around Analysis.')

#_______________________________________________________
st.header('Delay analysis :')


st.subheader('Get around delay analysis dataset :')
dataset = pd.read_csv('src/get_around_delay_analysis_clean.csv')
st.dataframe(dataset)


st.subheader('View of cars in late :')
dataset_late = dataset[dataset['delay_at_checkout_in_minutes'] > 0] # Just watch cars with late 
dataset_late = dataset_late[dataset_late['delay_at_checkout_in_minutes'] < 240] # + 4 hours, after it must be outlier 
fig = px.histogram(dataset_late, 'delay_at_checkout_in_minutes', color = 'checkin_type')
st.plotly_chart(fig, use_container_width=True)


st.subheader('View of cars in advance :')
dataset_early = dataset[dataset['delay_at_checkout_in_minutes'] < 0] # Just watch cars with early 
dataset_early = dataset_early[dataset_early['delay_at_checkout_in_minutes'] > -240] # + 4 hours, before it must be outlier 
fig = px.histogram(dataset_early, 'delay_at_checkout_in_minutes', color = 'checkin_type')
st.plotly_chart(fig, use_container_width=True)


st.subheader('Distribution Mobile and connect for early and late conductors :')
values_early = dataset_early.groupby('checkin_type')['delay_at_checkout_in_minutes'].sum().abs()
values_lates = dataset_late.groupby('checkin_type')['delay_at_checkout_in_minutes'].sum()
# Create subplots: use 'domain' type for Pie subplot
fig = make_subplots(rows=1, cols=2, specs=[[{'type':'domain'}, {'type':'domain'}]])
fig.add_trace(go.Pie(labels=values_early.keys(), values=values_early, name="Early Conductor"),
              1, 1)
fig.add_trace(go.Pie(labels=values_lates.keys(), values=values_lates, name="Late Conductor"),
             1, 2)
# Use `hole` to create a donut-like pie chart
fig.update_traces(hole=.4, hoverinfo="label+percent+name")
fig.update_layout(
    # Add annotations in the center of the donut pies.
    annotations=[dict(text='Early', x=0.19, y=0.5, font_size=20, showarrow=False),
                 dict(text='Late', x=0.80, y=0.5, font_size=20, showarrow=False)])
st.plotly_chart(fig, use_container_width=True)


st.subheader('Cars available between check in and check out after a delay of ...')
specs = np.repeat({'type':'domain'}, 5).tolist()
fig = make_subplots(rows=1, cols=5, specs=[specs])
for hours_cut in range(0,5):
    dataset_before = len( dataset[dataset['delay_at_checkout_in_minutes'] < (hours_cut*60)] )
    dataset_after = len( dataset[dataset['delay_at_checkout_in_minutes'] >= (hours_cut*60)] )
    fig.add_trace(go.Pie(labels=['Ready', 'Unvailable'], 
                         values=[dataset_before, dataset_after], 
                         name=f"{hours_cut} hour(s) between ck_in & check out"),
                         1, (hours_cut+1) )
fig.update_layout(
    # Add annotations in the center of the donut pies.
    annotations=[dict(text='Without', x=0.03, y=0.9, font_size=20, showarrow=False),
                 dict(text='After 1 hour', x=0.22, y=0.9, font_size=20, showarrow=False),
                 dict(text='After 2 hours', x=0.50, y=0.9, font_size=20, showarrow=False),
                 dict(text='After 3 hours', x=0.79, y=0.9, font_size=20, showarrow=False),
                 dict(text='After 4 hours', x=0.99, y=0.9, font_size=20, showarrow=False),
                ])
fig.update_traces(marker=dict(colors=['#636EFA', '#EF553B']))
st.plotly_chart(fig, use_container_width=True)


st.subheader('Cars available between check in and check out after a delay of ...')
specs = np.repeat({'type':'domain'}, 10).reshape(2, 5).tolist()
fig = make_subplots(rows=2, cols=5, specs=specs)
for hours_cut in range(0,5):
    dataset_h_cuts = dataset[dataset['checkin_type'] == 'mobile']
    dataset_before = len( dataset_h_cuts[dataset_h_cuts['delay_at_checkout_in_minutes'] < (hours_cut*60)] )
    dataset_after = len( dataset_h_cuts[dataset_h_cuts['delay_at_checkout_in_minutes'] >= (hours_cut*60)] )
    fig.add_trace(go.Pie(labels=['Ready', 'Unvailable'], 
                         values=[dataset_before, dataset_after], 
                         name=f"{hours_cut} hour(s) between ck_in & check out"),
                         1, (hours_cut+1) )
for hours_cut in range(0,5):
    dataset_h_cuts = dataset[dataset['checkin_type'] == 'connect']
    dataset_before = len( dataset_h_cuts[dataset_h_cuts['delay_at_checkout_in_minutes'] < (hours_cut*60)] )
    dataset_after = len( dataset_h_cuts[dataset_h_cuts['delay_at_checkout_in_minutes'] >= (hours_cut*60)] )
    fig.add_trace(go.Pie(labels=['Ready', 'Unvailable'], 
                         values=[dataset_before, dataset_after], 
                         name=f"{hours_cut} hour(s) between ck_in & check out"),
                         2, (hours_cut+1) )
fig.update_layout(
    # Add annotations in the center of the donut pies.
    annotations=[dict(text='Without', x=0.05, y=-0.1, font_size=12, showarrow=False),
                 dict(text='After 1 hour', x=0.24, y=-0.1, font_size=12, showarrow=False),
                 dict(text='After 2 hours', x=0.50, y=-0.1, font_size=12, showarrow=False),
                 dict(text='After 3 hours', x=0.76, y=-0.1, font_size=12, showarrow=False),
                 dict(text='After 4 hours', x=0.97, y=-0.1, font_size=12, showarrow=False),
                 dict(text='mobile', x=-0.05, y=0.95, font_size=20, showarrow=False,textangle=-90),
                 dict(text='connect', x=-0.05, y=0.05, font_size=20, showarrow=False, textangle=-90),
                ])
fig.update_traces(marker=dict(colors=['#636EFA', '#EF553B']))
st.plotly_chart(fig, use_container_width=True)



#_______________________________________________________
st.subheader('Pricing project :')
dataset = pd.read_csv('src/get_around_pricing_project_clean.csv')
st.dataframe(dataset)

st.subheader('Informations about columns of Pricing project dataset :')
dataset_plot = dataset
target = 'rental_price_per_day'

for column in dataset_plot.columns:
    if dataset_plot[column].dtypes == "object":
        # Quantitative Values
        fig = px.histogram(dataset_plot[column])
        fig.update_layout(title= f"{column.replace('_', ' ')}")
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    elif dataset_plot[column].dtypes == bool:
        # Bool Values 
        cat_data = dataset_plot.groupby(column)[target].sum()
        fig = px.bar(x=cat_data.index, y=cat_data, labels=dict(x="", y=""))
        fig.update_layout(title= f"{column.replace('_', ' ')}")
        st.plotly_chart(fig, use_container_width=True)

    else:
        # Qualitative Values
        fig = px.histogram(dataset_plot[column])
        fig.update_layout(title= f"{column.replace('_', ' ')}")
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

st.caption('This EDA page give some informations about Get Around datasets.')
st.caption('Github of the project on this page : [Bloc n°5](https://github.com/g0thier/Bloc-5)')
st.caption("Projet Bloc n°5 Jedha by [Gauthier Rammault](https://www.linkedin.com/in/gauthier-rammault/), the guy dreams to wanna be a real Data Scientist.")

# st.markdown("[How make my streamlit page](https://docs.streamlit.io/library/api-reference)")