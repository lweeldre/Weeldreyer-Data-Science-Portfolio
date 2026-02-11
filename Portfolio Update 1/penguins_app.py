import streamlit as st
import pandas as pd

# Load the CSV file:
df = pd.read_csv("data/penguins.csv")

# Expanding the configuration to use the whole width of the screen:
st.set_page_config(page_title = "Penguin Analytics", layout = "wide", page_icon = "🐧")

# Markdown Hashtags:
st.title("Penguin Species Statistics")
st.subheader("Dataset Specifics")

# Produce the Data set in its simplest form:
st.write("Here's our data set")
st.dataframe(df)

# Creating pull-down bar for each species:
species = st.selectbox("Select a Species", df["species"].unique(), index = None)
filtered_df = df[df["species"] == species]

# Producing Statistics for label selected in pull-down bar:
st.write(f"{species} Statistics")
st.dataframe(filtered_df)

# Description for data Table #2:
st.write(f"This data table allows you observe specific statistics after selecting a certain species of penguins from the pull-down bar.")

# Creating Interactive experience to explore the data:
st.title("Penguin Species Explorer")

# Creating a sidebar for filtering:
st.sidebar.header("Filter Options:")
selected_species = st.sidebar.multiselect("Select Species", options = df['species'].unique(), default = df['species'].unique())

# Filtering the dataframe:
filtered_df = df[df['species'].isin(selected_species)]

# Metrics Section:
col1,col2, col3 = st.columns(3)
col1.metric("Total Penguins", len(filtered_df))
col2.metric("Avg Body Mass (g)", int(filtered_df['body_mass_g'].mean()))
col3.metric("Avg Bill Length (mm)", round(filtered_df['bill_length_mm'].mean(), 1))

# Show Table:
st.subheader("Data Overview")
st.dataframe(filtered_df, use_container_width=True)

# Description for Data Table #3:
st.write("This data table allows you to observe the total amount of penguins, Average Body Mass (g), and Average Bill Length (mm) using the filter optoins on the left side of the screen. The filter options allow you to observe multile species at once. ")

# Creating a Scatter plot:
st.title("Penguin Measurement Correlations")

# Select boxes for x and y axis:
numeric_cols = ['bill_length_mm', 'bill_depth_mm','flipper_length_mm', 'body_mass_g']

col1, col2 = st.columns(2)
with col1:
    x_axis = st.selectbox("select X-axis", numeric_cols, index = 0)
with col2:
    y_axis = st.selectbox("Select Y-axis", numeric_cols, index = 1)

# Scatter Chart colored by Species:
st.write(f"### Plotting {x_axis} vs {y_axis}")
st.scatter_chart(data = df, x = x_axis, y = y_axis, color = "species", size = "body_mass_g")

# Description for Data set #4:
st.write("This data table presents a cluster that illustrates the differences and similarities in Bill Length (mm) among the three different species of penguins.")

# Creating a Bar Chart:
st.title("Population Distributions")

# Tabs for Organization:
tab1, tab2 = st.tabs(["Species Count by Ireland", "Average Body Mass"])

with tab1:
    st.subheader("Penguins per Island")
    # Group Data for Bar Chart:
    island_counts = df.groupby(['island', 'species']).size().reset_index(name = 'count')
    st.bar_chart(island_counts, x = "island", y = "count", color = "species", stack  = False)

with tab2:
    st.subheader("Average Body Mass per Species")
    # Using a radio button to toggle a view:
    show_raw = st.toggle("Show raw numbers")
    avg_mass = df.groupby('species')['body_mass_g'].mean()

    st.bar_chart(avg_mass)
    if show_raw:
        st.write(avg_mass)

#Description for Data set #5:
st.write("This data table presents a bar chart that illustrates the population distribution of the different species of penguins by island. this bar chart also presents the average body mass for each species.")