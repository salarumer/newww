# pylint: disable=broad-exception-caught,invalid-name

import time
import io
import base64
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
import json

from google import genai
from google.cloud import bigquery
from google.genai.types import FunctionDeclaration, GenerateContentConfig, Part, Tool
import streamlit as st
import pandas as pd

BIGQUERY_DATASET_ID = "dataset1"
MODEL_ID = "gemini-1.5-pro"
LOCATION = "us-central1"

# Set Matplotlib style for better looking static charts
matplotlib.style.use('ggplot')
plt.rcParams['figure.figsize'] = (10, 7)
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.labelsize'] = 12

list_datasets_func = FunctionDeclaration(
    name="list_datasets",
    description="Get a list of datasets that will help answer the user's question",
    parameters={
        "type": "object",
        "properties": {},
    },
)

list_tables_func = FunctionDeclaration(
    name="list_tables",
    description="List tables in a dataset that will help answer the user's question",
    parameters={
        "type": "object",
        "properties": {
            "dataset_id": {
                "type": "string",
                "description": "Dataset ID to fetch tables from.",
            }
        },
        "required": [
            "dataset_id",
        ],
    },
)

get_table_func = FunctionDeclaration(
    name="get_table",
    description="Get information about a table, including the description, schema, and number of rows that will help answer the user's question. Always use the fully qualified dataset and table names.",
    parameters={
        "type": "object",
        "properties": {
            "table_id": {
                "type": "string",
                "description": "Fully qualified ID of the table to get information about",
            }
        },
        "required": [
            "table_id",
        ],
    },
)

sql_query_func = FunctionDeclaration(
    name="sql_query",
    description="Get information from data in BigQuery using SQL queries",
    parameters={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "SQL query on a single line that will help give quantitative answers to the user's question when run on a BigQuery dataset and table. In the SQL query, always use the fully qualified dataset and table names.",
            }
        },
        "required": [
            "query",
        ],
    },
)

create_pie_chart_func = FunctionDeclaration(
    name="create_pie_chart",
    description="Create an interactive pie chart visualization based on query results",
    parameters={
        "type": "object",
        "properties": {
            "title": {
                "type": "string",
                "description": "Title for the pie chart",
            },
            "labels_column": {
                "type": "string",
                "description": "Column name to use for pie chart labels",
            },
            "values_column": {
                "type": "string",
                "description": "Column name to use for pie chart values",
            },
            "query": {
                "type": "string",
                "description": "SQL query that returns data suitable for a pie chart (typically two columns: one for categories/labels and one for numeric values)",
            }
        },
        "required": [
            "title",
            "labels_column",
            "values_column",
            "query",
        ],
    },
)

create_bar_chart_func = FunctionDeclaration(
    name="create_bar_chart",
    description="Create an interactive bar chart visualization based on query results",
    parameters={
        "type": "object",
        "properties": {
            "title": {
                "type": "string",
                "description": "Title for the bar chart",
            },
            "labels_column": {
                "type": "string",
                "description": "Column name to use for bar chart labels",
            },
            "values_column": {
                "type": "string",
                "description": "Column name to use for bar chart values",
            },
            "query": {
                "type": "string",
                "description": "SQL query that returns data suitable for a bar chart (typically two columns: one for categories/labels and one for numeric values)",
            }
        },
        "required": [
            "title",
            "labels_column",
            "values_column",
            "query",
        ],
    },
)

sql_query_tool = Tool(
    function_declarations=[
        list_datasets_func,
        list_tables_func,
        get_table_func,
        sql_query_func,
        create_pie_chart_func,
        create_bar_chart_func,
    ],
)

client = genai.Client(vertexai=True, location=LOCATION)

st.set_page_config(
    page_title="SQL Talk with BigQuery",
    page_icon="vertex-ai.png",
    layout="wide",
)
# Set background color to white and improve UI
st.markdown(
    """
    <style>
    .stApp {
        background-color: white !important;  /* Pure white background */
        color: black !important;  /* Black text */
    }
    .stMarkdown, .stTextInput, .stTextArea, .stButton {
        color: black !important;
    }
    .chart-download-btn {
        background-color: #f0f2f6;
        border-radius: 4px;
        padding: 8px;
        margin-top: 10px;
        margin-bottom: 10px;
    }
    .stExpander {
        border-radius: 8px;
        border: 1px solid #e0e0e0;
    }
    .plot-container {
        border-radius: 8px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
        padding: 10px;
        background-color: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)

col1, col2 = st.columns([8, 1])
with col1:
    st.title("SQL Talk with BigQuery")
with col2:
    st.image("vertex-ai.png")

st.subheader("Powered by Function Calling in Gemini")

st.markdown(
    "[Source Code](https://github.com/GoogleCloudPlatform/generative-ai/tree/main/gemini/function-calling/sql-talk-app/)   •   [Documentation](https://cloud.google.com/vertex-ai/docs/generative-ai/multimodal/function-calling)   •   [Codelab](https://codelabs.developers.google.com/codelabs/gemini-function-calling)   •   [Sample Notebook](https://github.com/GoogleCloudPlatform/generative-ai/blob/main/gemini/function-calling/intro_function_calling.ipynb)"
)

with st.expander("Sample prompts", expanded=True):
    st.write(
        """
        - What kind of information is in this database?
        - What percentage of orders are returned?
        - How is inventory distributed across our regional distribution centers? Show as pie chart.
        - Do customers typically place more than one order?
        - Which product categories have the highest profit margins? Visualize with a pie chart.
    """
    )

if "messages" not in st.session_state:
    st.session_state.messages = []

if "pie_charts" not in st.session_state:
    st.session_state.pie_charts = {}

if "bar_charts" not in st.session_state:
    st.session_state.bar_charts = {}

# Function to create PDF from dataframe and chart
def create_pdf(chart_data, chart_type):
    buf = io.BytesIO()
    
    with PdfPages(buf) as pdf:
        # First page with chart
        fig = plt.figure(figsize=(10, 7))
        if chart_type == "pie":
            plt.imshow(Image.open(io.BytesIO(chart_data["image_bytes"])))
        else:
            plt.imshow(Image.open(io.BytesIO(chart_data["image_bytes"])))
        plt.axis('off')
        pdf.savefig(fig)
        plt.close(fig)
        
        # Second page with data table
        fig, ax = plt.subplots(figsize=(10, 7))
        df = pd.DataFrame(chart_data["data"])
        ax.axis('tight')
        ax.axis('off')
        table = ax.table(cellText=df.values, colLabels=df.columns, loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.2)
        plt.title(f"Data for: {chart_data['title']}")
        pdf.savefig(fig)
        plt.close(fig)
    
    buf.seek(0)
    return buf.getvalue()

# Display messages and charts
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"].replace("$", r"\$"))  # noqa: W605
        
        # Display pie chart if available
        if "chart_id" in message and message["chart_id"] in st.session_state.pie_charts:
            chart_data = st.session_state.pie_charts[message["chart_id"]]
            with st.container(border=True, class_="plot-container"):
                # Display the interactive chart
                if "plotly_fig" in chart_data:
                    st.plotly_chart(chart_data["plotly_fig"], use_container_width=True)
                else:
                    st.image(chart_data["image"])
                
                # Create download options with dropdown
                col1, col2, col3 = st.columns([1, 1, 1])
                with col1:
                    st.download_button(
                        label="Download PNG",
                        data=chart_data["image_bytes"],
                        file_name=f"{chart_data['title'].replace(' ', '_')}.png",
                        mime="image/png"
                    )
                with col2:
                    # Convert PNG to JPG for download
                    img = Image.open(io.BytesIO(chart_data["image_bytes"]))
                    jpg_buf = io.BytesIO()
                    img.convert('RGB').save(jpg_buf, format='JPEG')
                    jpg_buf.seek(0)
                    
                    st.download_button(
                        label="Download JPG",
                        data=jpg_buf.getvalue(),
                        file_name=f"{chart_data['title'].replace(' ', '_')}.jpg",
                        mime="image/jpeg"
                    )
                with col3:
                    # Create PDF with chart and data
                    pdf_bytes = create_pdf(chart_data, "pie")
                    
                    st.download_button(
                        label="Download PDF",
                        data=pdf_bytes,
                        file_name=f"{chart_data['title'].replace(' ', '_')}.pdf",
                        mime="application/pdf"
                    )
                
        if "chart_id" in message and message["chart_id"] in st.session_state.bar_charts:
            chart_data = st.session_state.bar_charts[message["chart_id"]]
            with st.container(border=True, class_="plot-container"):
                # Display the interactive chart
                if "plotly_fig" in chart_data:
                    st.plotly_chart(chart_data["plotly_fig"], use_container_width=True)
                else:
                    st.image(chart_data["image"])
                
                # Create download options with dropdown
                col1, col2, col3 = st.columns([1, 1, 1])
                with col1:
                    st.download_button(
                        label="Download PNG",
                        data=chart_data["image_bytes"],
                        file_name=f"{chart_data['title'].replace(' ', '_')}.png",
                        mime="image/png"
                    )
                with col2:
                    # Convert PNG to JPG for download
                    img = Image.open(io.BytesIO(chart_data["image_bytes"]))
                    jpg_buf = io.BytesIO()
                    img.convert('RGB').save(jpg_buf, format='JPEG')
                    jpg_buf.seek(0)
                    
                    st.download_button(
                        label="Download JPG",
                        data=jpg_buf.getvalue(),
                        file_name=f"{chart_data['title'].replace(' ', '_')}.jpg",
                        mime="image/jpeg"
                    )
                with col3:
                    # Create PDF with chart and data
                    pdf_bytes = create_pdf(chart_data, "bar")
                    
                    st.download_button(
                        label="Download PDF",
                        data=pdf_bytes,
                        file_name=f"{chart_data['title'].replace(' ', '_')}.pdf",
                        mime="application/pdf"
                    )
            
        try:
            with st.expander("Function calls, parameters, and responses"):
                st.markdown(message["backend_details"])
        except KeyError:
            pass

if prompt := st.chat_input("Ask me about information in the database..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        chat = client.chats.create(
            model=MODEL_ID,
            config=GenerateContentConfig(temperature=0, tools=[sql_query_tool]),
        )
        client = bigquery.Client()

        prompt += """
            Please give a concise, high-level summary followed by detail in
            plain language about where the information in your response is
            coming from in the database. Only use information that you learn
            from BigQuery, do not make up information. If the user's query mentions a visualization, 
            chart, or specifically a pie chart, use the create_pie_chart function to generate a pie chart 
            of the results or if specifically a bar chart, use the create_bar_chart function to generate a bar chart 
            of the results.
            """

        try:
            response = chat.send_message(prompt)
            response = response.candidates[0].content.parts[0]

            print(response)

            api_requests_and_responses = []
            backend_details = ""
            chart_id = None

            function_calling_in_process = True
            while function_calling_in_process:
                try:
                    params = {}
                    for key, value in response.function_call.args.items():
                        params[key] = value

                    print(response.function_call.name)
                    print(params)

                    if response.function_call.name == "list_datasets":
                        api_response = client.list_datasets()
                        api_response = BIGQUERY_DATASET_ID
                        api_requests_and_responses.append(
                            [response.function_call.name, params, api_response]
                        )

                    if response.function_call.name == "list_tables":
                        api_response = client.list_tables(params["dataset_id"])
                        api_response = str([table.table_id for table in api_response])
                        api_requests_and_responses.append(
                            [response.function_call.name, params, api_response]
                        )

                    if response.function_call.name == "get_table":
                        api_response = client.get_table(params["table_id"])
                        api_response = api_response.to_api_repr()
                        api_requests_and_responses.append(
                            [
                                response.function_call.name,
                                params,
                                [
                                    str(api_response.get("description", "")),
                                    str(
                                        [
                                            column["name"]
                                            for column in api_response["schema"][
                                                "fields"
                                            ]
                                        ]
                                    ),
                                ],
                            ]
                        )
                        api_response = str(api_response)

                    if response.function_call.name == "sql_query":
                        job_config = bigquery.QueryJobConfig(
                            maximum_bytes_billed=100000000
                        )  # Data limit per query job
                        try:
                            cleaned_query = (
                                params["query"]
                                .replace("\\n", " ")
                                .replace("\n", "")
                                .replace("\\", "")
                            )
                            query_job = client.query(
                                cleaned_query, job_config=job_config
                            )
                            api_response = query_job.result()
                            api_response = str([dict(row) for row in api_response])
                            api_response = api_response.replace("\\", "").replace(
                                "\n", ""
                            )
                            api_requests_and_responses.append(
                                [response.function_call.name, params, api_response]
                            )
                        except Exception as e:
                            error_message = f"""
                            We're having trouble running this SQL query. This
                            could be due to an invalid query or the structure of
                            the data. Try rephrasing your question to help the
                            model generate a valid query. Details:

                            {str(e)}"""
                            st.error(error_message)
                            api_response = error_message
                            api_requests_and_responses.append(
                                [response.function_call.name, params, api_response]
                            )
                            st.session_state.messages.append(
                                {
                                    "role": "assistant",
                                    "content": error_message,
                                }
                            )

                    if response.function_call.name == "create_bar_chart":
                        job_config = bigquery.QueryJobConfig(
                            maximum_bytes_billed=100000000
                        )
                        try:
                            # Run the query to get data for the bar chart
                            cleaned_query = (
                                params["query"]
                                .replace("\\n", " ")
                                .replace("\n", "")
                                .replace("\\", "")
                            )
                            query_job = client.query(
                                cleaned_query, job_config=job_config
                            )
                            query_results = query_job.result()
                            
                            # Convert to DataFrame
                            df = query_results.to_dataframe()
                            
                            # Create the static bar chart for image format
                            fig, ax = plt.subplots(figsize=(10, 7))
                            
                            # Add some style
                            colors = plt.cm.viridis(range(len(df)))
                            
                            df.plot.bar(
                                x=params["labels_column"],
                                y=params["values_column"],
                                ax=ax,
                                rot=45,
                                legend=False,
                                color=colors
                            )
                            ax.set_title(params["title"], fontweight='bold', fontsize=16)
                            ax.grid(axis='y', linestyle='--', alpha=0.7)
                            
                            # Add labels with values on the bars
                            for i, v in enumerate(df[params["values_column"]]):
                                ax.text(i, v + 0.1, f'{v:.1f}', ha='center', fontweight='bold')
                                
                            plt.tight_layout()
                            
                            # Save chart to memory for display
                            buf = io.BytesIO()
                            fig.savefig(buf, format='png', bbox_inches='tight', dpi=300)
                            buf.seek(0)
                            img_bytes = buf.getvalue()
                            img_b64 = base64.b64encode(img_bytes).decode()
                            img_src = f"data:image/png;base64,{img_b64}"
                            
                            # Create interactive Plotly bar chart
                            plotly_fig = px.bar(
                                df, 
                                x=params["labels_column"], 
                                y=params["values_column"],
                                title=params["title"],
                                labels={
                                    params["labels_column"]: params["labels_column"].replace('_', ' ').title(),
                                    params["values_column"]: params["values_column"].replace('_', ' ').title()
                                },
                                color_discrete_sequence=px.colors.qualitative.Vivid,
                                template="plotly_white"
                            )
                            
                            # Enhance Plotly styling
                            plotly_fig.update_layout(
                                title_font=dict(size=20, family="Arial Black", color="#333333"),
                                font=dict(family="Arial", size=14),
                                hoverlabel=dict(bgcolor="white", font_size=14, font_family="Arial"),
                                plot_bgcolor='rgba(240, 240, 240, 0.8)',
                                xaxis=dict(
                                    title_font=dict(size=16),
                                    tickangle=-45
                                ),
                                yaxis=dict(
                                    title_font=dict(size=16),
                                    gridcolor='rgba(200, 200, 200, 0.5)'
                                ),
                                bargap=0.2,
                                margin=dict(t=80, b=80, l=60, r=40),
                            )
                            
                            # Add value labels on hover
                            plotly_fig.update_traces(
                                texttemplate='%{y:.1f}',
                                textposition='outside',
                                hovertemplate='<b>%{x}</b><br>%{y:.2f}<extra></extra>'
                            )
                            
                            # Generate a unique ID for this chart
                            chart_id = f"chart_{len(st.session_state.bar_charts)}"
                            
                            # Save chart data in session state
                            st.session_state.bar_charts[chart_id] = {
                                "image": img_src,
                                "image_bytes": img_bytes,
                                "title": params["title"],
                                "data": df.to_dict(),
                                "plotly_fig": plotly_fig
                            }
                            
                            # Display the chart
                            chart_placeholder = st.empty()
                            
                            api_response = {
                                "success": True,
                                "message": f"Created interactive bar chart titled '{params['title']}' with {len(df)} data points.",
                                "chart_id": chart_id
                            }
                            api_response = str(api_response)
                            api_requests_and_responses.append(
                                [response.function_call.name, params, api_response]
                            )
                            
                        except Exception as e:
                            error_message = f"""
                            We're having trouble creating the bar chart. This
                            could be due to an invalid query or unsuitable data structure.
                            Try rephrasing your question. Details:

                            {str(e)}"""
                            st.error(error_message)
                            api_response = error_message
                            api_requests_and_responses.append(
                                [response.function_call.name, params, api_response]
                            )

                    if response.function_call.name == "create_pie_chart":
                        job_config = bigquery.QueryJobConfig(
                            maximum_bytes_billed=100000000
                        )
                        try:
                            # Run the query to get data for the pie chart
                            cleaned_query = (
                                params["query"]
                                .replace("\\n", " ")
                                .replace("\n", "")
                                .replace("\\", "")
                            )
                            query_job = client.query(
                                cleaned_query, job_config=job_config
                            )
                            query_results = query_job.result()
                            
                            # Convert to DataFrame
                            df = query_results.to_dataframe()
                            
                            # Create the static pie chart for image format
                            fig, ax = plt.subplots(figsize=(10, 7))
                            
                            # Use a better colormap
                            cmap = plt.cm.viridis
                            colors = cmap(range(len(df)))
                            
                            # Create a visually appealing pie chart
                            wedges, texts, autotexts = ax.pie(
                                df[params["values_column"]],
                                labels=None,  # We'll add a legend instead
                                autopct='%1.1f%%',
                                startangle=90,
                                shadow=True,
                                colors=colors,
                                explode=[0.05] * len(df),  # Slightly explode all slices
                                wedgeprops=dict(width=0.5, edgecolor='w'),  # Create a donut chart
                                textprops=dict(fontsize=12, fontweight='bold', color='white')
                            )
                            
                            # Add a circle at the center to create a donut chart
                            centre_circle = plt.Circle((0, 0), 0.25, fc='white')
                            ax.add_patch(centre_circle)
                            
                            # Add title with better styling
                            ax.set_title(params["title"], fontsize=16, fontweight='bold', pad=20)
                            
                            # Add a legend with category labels
                            ax.legend(
                                wedges, 
                                df[params["labels_column"]], 
                                title=params["labels_column"].replace('_', ' ').title(),
                                loc="center left", 
                                bbox_to_anchor=(1, 0, 0.5, 1)
                            )
                            
                            plt.tight_layout()
                            
                            # Save chart to memory for display
                            buf = io.BytesIO()
                            fig.savefig(buf, format='png', bbox_inches='tight', dpi=300)
                            buf.seek(0)
                            img_bytes = buf.getvalue()
                            img_b64 = base64.b64encode(img_bytes).decode()
                            img_src = f"data:image/png;base64,{img_b64}"
                            
                            # Create interactive Plotly pie chart
                            plotly_fig = px.pie(
                                df, 
                                values=params["values_column"], 
                                names=params["labels_column"],
                                title=params["title"],
                                color_discrete_sequence=px.colors.qualitative.Vivid,
                                template="plotly_white",
                                hole=0.4,  # Create a donut chart
                            )
                            
                            # Enhance Plotly styling
                            plotly_fig.update_layout(
                                title_font=dict(size=20, family="Arial Black", color="#333333"),
                                font=dict(family="Arial", size=14),
                                hoverlabel=dict(bgcolor="white", font_size=14, font_family="Arial"),
                                showlegend=True,
                                legend=dict(
                                    orientation="h",
                                    yanchor="bottom",
                                    y=-0.2,
                                    xanchor="center",
                                    x=0.5
                                ),
                                margin=dict(t=80, b=80, l=20, r=20),
                                annotations=[dict(
                                    text='Interactive<br>Chart',
                                    showarrow=False,
                                    font=dict(size=16)
                                )]
                            )
                            
                            # Add pull to slices and improve hover info
                            plotly_fig.update_traces(
                                pull=[0.03] * len(df),
                                textposition='inside',
                                textinfo='percent+label',
                                hovertemplate='<b>%{label}</b><br>%{value:.2f} (%{percent})<extra></extra>'
                            )
                            
                            # Generate a unique ID for this chart
                            chart_id = f"chart_{len(st.session_state.pie_charts)}"
                            
                            # Save chart data in session state
                            st.session_state.pie_charts[chart_id] = {
                                "image": img_src,
                                "image_bytes": img_bytes,
                                "title": params["title"],
                                "data": df.to_dict(),
                                "plotly_fig": plotly_fig
                            }
                            
                            # Display the chart
                            chart_placeholder = st.empty()
                            
                            api_response = {
                                "success": True,
                                "message": f"Created interactive pie chart titled '{params['title']}' with {len(df)} data points.",
                                "chart_id": chart_id
                            }
                            api_response = str(api_response)
                            api_requests_and_responses.append(
                                [response.function_call.name, params, api_response]
                            )
                            
                        except Exception as e:
                            error_message = f"""
                            We're having trouble creating the pie chart. This
                            could be due to an invalid query or unsuitable data structure.
                            Try rephrasing your question. Details:

                            {str(e)}"""
                            st.error(error_message)
                            api_response = error_message
                            api_requests_and_responses.append(
                                [response.function_call.name, params, api_response]
                            )

                    print(api_response)

                    response = chat.send_message(
                        Part.from_function_response(
                            name=response.function_call.name,
                            response={
                                "content": api_response,
                            },
                        ),
                    )
                    response = response.candidates[0].content.parts[0]

                    backend_details += "- Function call:\n"
                    backend_details += (
                        "   - Function name: "
                        + str(api_requests_and_responses[-1][0])
                        + ""
                    )
                    backend_details += "\n\n"
                    backend_details += (
                        "   - Function parameters: "
                        + str(api_requests_and_responses[-1][1])
                        + ""
                    )
                    backend_details += "\n\n"
                    backend_details += (
                        "   - API response: "
                        + str(api_requests_and_responses[-1][2])
                        + ""
                    )
                    backend_details += "\n\n"
                    with message_placeholder.container():
                        st.markdown(backend_details)

                except AttributeError:
                    function_calling_in_process = False

            # Store the final response in session state
            st.session_state.messages.append({
                "role": "assistant",
                "content": response.text,
                "backend_details": backend_details,
                "chart_id": chart_id
            })

            # Display final response
            message_placeholder.empty()
            message_placeholder.markdown(response.text.replace("$", r"\$"))  # no
# Display the chart if applicable
            if chart_id is not None:
                if chart_id in st.session_state.pie_charts:
                    chart_data = st.session_state.pie_charts[chart_id]
                    with st.container(border=True, class_="plot-container"):
                        # Display the interactive chart
                        if "plotly_fig" in chart_data:
                            st.plotly_chart(chart_data["plotly_fig"], use_container_width=True)
                        else:
                            st.image(chart_data["image"])
                        
                        # Create download options with dropdown
                        col1, col2, col3 = st.columns([1, 1, 1])
                        with col1:
                            st.download_button(
                                label="Download PNG",
                                data=chart_data["image_bytes"],
                                file_name=f"{chart_data['title'].replace(' ', '_')}.png",
                                mime="image/png"
                            )
                        with col2:
                            # Convert PNG to JPG for download
                            img = Image.open(io.BytesIO(chart_data["image_bytes"]))
                            jpg_buf = io.BytesIO()
                            img.convert('RGB').save(jpg_buf, format='JPEG')
                            jpg_buf.seek(0)
                            
                            st.download_button(
                                label="Download JPG",
                                data=jpg_buf.getvalue(),
                                file_name=f"{chart_data['title'].replace(' ', '_')}.jpg",
                                mime="image/jpeg"
                            )
                        with col3:
                            # Create PDF with chart and data
                            pdf_bytes = create_pdf(chart_data, "pie")
                            
                            st.download_button(
                                label="Download PDF",
                                data=pdf_bytes,
                                file_name=f"{chart_data['title'].replace(' ', '_')}.pdf",
                                mime="application/pdf"
                            )
                if chart_id in st.session_state.bar_charts:
                    chart_data = st.session_state.bar_charts[chart_id]
                    with st.container(border=True, class_="plot-container"):
                        # Display the interactive chart
                        if "plotly_fig" in chart_data:
                            st.plotly_chart(chart_data["plotly_fig"], use_container_width=True)
                        else:
                            st.image(chart_data["image"])
                        
                        # Create download options with dropdown
                        col1, col2, col3 = st.columns([1, 1, 1])
                        with col1:
                            st.download_button(
                                label="Download PNG",
                                data=chart_data["image_bytes"],
                                file_name=f"{chart_data['title'].replace(' ', '_')}.png",
                                mime="image/png"
                            )
                        with col2:
                            # Convert PNG to JPG for download
                            img = Image.open(io.BytesIO(chart_data["image_bytes"]))
                            jpg_buf = io.BytesIO()
                            img.convert('RGB').save(jpg_buf, format='JPEG')
                            jpg_buf.seek(0)
                            
                            st.download_button(
                                label="Download JPG",
                                data=jpg_buf.getvalue(),
                                file_name=f"{chart_data['title'].replace(' ', '_')}.jpg",
                                mime="image/jpeg"
                            )
                        with col3:
                            # Create PDF with chart and data
                            pdf_bytes = create_pdf(chart_data, "bar")
                            
                            st.download_button(
                                label="Download PDF",
                                data=pdf_bytes,
                                file_name=f"{chart_data['title'].replace(' ', '_')}.pdf",
                                mime="application/pdf"
                            )

        except Exception as e:
            error_message = f"Error: {str(e)}"
            st.error(error_message)
            st.session_state.messages.append(
                {"role": "assistant", "content": error_message}
            )
