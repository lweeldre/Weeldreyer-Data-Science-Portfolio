import streamlit as st

# markdown Hashtag
st.title("Hello Streamlit!")

st.write("This is my first Steamlit App.")

if st.button("Click me!"):
    st.write("You clicked the button")
else:
    st.write("Click the button and see what happens")

# Load our CSV file
import pandas as pd

st.subheader("exploring our dataset")

# Load the CSV file
df = pd.read_csv("data/sample_data-1.csv")

st.write("Here's our data")
st.dataframe(df)

city = st.selectbox("Select a City", df["City"].unique(), index = None)
filtered_df = df[df["City"] == city]

st.write(f"people in {city}")
st.dataframe(filtered_df)

# Add bar chart
st.bar_chart(df["Salary"])

import seaborn as sns

box_plot1 = sns.boxplot(x = df["City"], y = df["Salary"])

st.pyplot(box_plot1.get_figure())

