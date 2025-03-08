# pylint: disable=broad-exception-caught,invalid-name

import time
import io
import base64
import matplotlib.pyplot as plt

from google import genai
from google.cloud import bigquery
from google.genai.types import FunctionDeclaration, GenerateContentConfig, Part, Tool
import streamlit as st
import pandas as pd

BIGQUERY_DATASET_ID = "dataset1"
MODEL_ID = "gemini-1.5-pro"
LOCATION = "us-central1"

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
    description="Create a pie chart visualization based on query results",
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
    description="Create a bar chart visualization based on query results",
    parameters={
        "type": "object",
        "properties": {
            "title": {
                "type": "string",
                "description": "Title for the bar chart",
            },
            "x_column": {
                "type": "string",
                "description": "Column name to use for x-axis categories",
            },
            "y_column": {
                "type": "string",
                "description": "Column name to use for y-axis values",
            },
            "query": {
                "type": "string",
                "description": "SQL query that returns data suitable for a bar chart (typically two columns: one for categories and one for numeric values)",
            },
            "horizontal": {
                "type": "boolean",
                "description": "Whether to create a horizontal bar chart (true) or vertical bar chart (false)",
                "default": False,
            }
        },
        "required": [
            "title",
            "x_column",
            "y_column",
            "query",
        ],
    },
)

create_line_chart_func = FunctionDeclaration(
    name="create_line_chart",
    description="Create a line chart visualization based on query results",
    parameters={
        "type": "object",
        "properties": {
            "title": {
                "type": "string",
                "description": "Title for the line chart",
            },
            "x_column": {
                "type": "string",
                "description": "Column name to use for x-axis (typically time or sequential data)",
            },
            "y_columns": {
                "type": "array",
                "items": {
                    "type": "string"
                },
                "description": "Column names to use for y-axis values (can be multiple for multi-line charts)",
            },
            "query": {
                "type": "string",
                "description": "SQL query that returns data suitable for a line chart (typically date/time column and one or more numeric columns)",
            }
        },
        "required": [
            "title",
            "x_column",
            "y_columns",
            "query",
        ],
    },
)

create_scatter_plot_func = FunctionDeclaration(
    name="create_scatter_plot",
    description="Create a scatter plot visualization based on query results",
    parameters={
        "type": "object",
        "properties": {
            "title": {
                "type": "string",
                "description": "Title for the scatter plot",
            },
            "x_column": {
                "type": "string",
                "description": "Column name to use for x-axis values",
            },
            "y_column": {
                "type": "string",
                "description": "Column name to use for y-axis values",
            },
            "query": {
                "type": "string",
                "description": "SQL query that returns data suitable for a scatter plot (typically two numeric columns)",
            },
            "color_column": {
                "type": "string",
                "description": "Optional column name to use for point colors to show categories",
            }
        },
        "required": [
            "title",
            "x_column",
            "y_column",
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
        create_line_chart_func,
        create_scatter_plot_func,
    ],
)

client = genai.Client(vertexai=True, location=LOCATION)

st.set_page_config(
    page_title="SQL Talk with BigQuery",
    page_icon="vertex-ai.png",
    layout="wide",
)
# Set background color to white
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
        - Show me monthly sales trends as a line chart.
        - Compare product categories by revenue using a bar chart.
        - Create a scatter plot of order value vs. customer age.
    """
    )

if "messages" not in st.session_state:
    st.session_state.messages = []

if "charts" not in st.session_state:
    st.session_state.charts = {}

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"].replace("$", r"\$"))  # noqa: W605
        
        # Display chart if available
        if "chart_id" in message and message["chart_id"] in st.session_state.charts:
            chart_data = st.session_state.charts[message["chart_id"]]
            st.image(chart_data["image"])
            st.download_button(
                label=f"Download {chart_data['chart_type'].title()} Chart",
                data=chart_data["image_bytes"],
                file_name=f"{chart_data['title'].replace(' ', '_')}.png",
                mime="image/png"
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
            from BigQuery, do not make up information. 
            
            If the user's query mentions visualization or data exploration, choose the most appropriate chart type:
            - Use pie charts for showing proportions and distribution (especially with categories)
            - Use bar charts for comparing values across categories
            - Use line charts for showing trends over time
            - Use scatter plots for showing relationships between two numeric variables
            
            Based on the data and question, select the most appropriate visualization function.
            """

        try:
            response = chat.send_message(prompt)
            response = response.candidates[0].content.parts[0]

            print(response)

            api_requests_and_responses = []
            backend_details = ""
            chart_id = None
            chart_type = None

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

                    # Create Pie Chart function
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
                            
                            # Create the pie chart
                            fig, ax = plt.subplots(figsize=(10, 7))
                            df.plot.pie(
                                y=params["values_column"],
                                labels=df[params["labels_column"]],
                                ax=ax,
                                autopct='%1.1f%%',
                                startangle=90,
                                shadow=False,
                            )
                            ax.set_title(params["title"])
                            ax.set_ylabel('')  # Hide y-label
                            
                            # Save chart to memory for display
                            buf = io.BytesIO()
                            fig.savefig(buf, format='png', bbox_inches='tight')
                            buf.seek(0)
                            img_bytes = buf.getvalue()
                            img_b64 = base64.b64encode(img_bytes).decode()
                            img_src = f"data:image/png;base64,{img_b64}"
                            
                            # Generate a unique ID for this chart
                            chart_id = f"chart_{len(st.session_state.charts)}"
                            chart_type = "pie chart"
                            
                            # Save chart data in session state
                            st.session_state.charts[chart_id] = {
                                "image": img_src,
                                "image_bytes": img_bytes,
                                "title": params["title"],
                                "data": df.to_dict(),
                                "chart_type": chart_type
                            }
                            
                            api_response = {
                                "success": True,
                                "message": f"Created pie chart titled '{params['title']}' with {len(df)} data points.",
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

                    # Create Bar Chart function
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
                            
                            # Create the bar chart
                            fig, ax = plt.subplots(figsize=(12, 8))
                            
                            # Check if horizontal bar chart is requested
                            is_horizontal = params.get("horizontal", False)
                            
                            if is_horizontal:
                                ax.barh(df[params["x_column"]], df[params["y_column"]])
                                ax.set_xlabel(params["y_column"])
                                ax.set_ylabel(params["x_column"])
                            else:
                                ax.bar(df[params["x_column"]], df[params["y_column"]])
                                ax.set_xlabel(params["x_column"])
                                ax.set_ylabel(params["y_column"])
                            
                            ax.set_title(params["title"])
                            
                            # Add data labels
                            for i, v in enumerate(df[params["y_column"]]):
                                if is_horizontal:
                                    ax.text(v + max(df[params["y_column"]]) * 0.01, i, f"{v:,.2f}", 
                                           va='center')
                                else:
                                    ax.text(i, v + max(df[params["y_column"]]) * 0.01, f"{v:,.2f}", 
                                           ha='center')
                            
                            plt.xticks(rotation=45 if not is_horizontal else 0)
                            plt.tight_layout()
                            
                            # Save chart to memory for display
                            buf = io.BytesIO()
                            fig.savefig(buf, format='png', bbox_inches='tight')
                            buf.seek(0)
                            img_bytes = buf.getvalue()
                            img_b64 = base64.b64encode(img_bytes).decode()
                            img_src = f"data:image/png;base64,{img_b64}"
                            
                            # Generate a unique ID for this chart
                            chart_id = f"chart_{len(st.session_state.charts)}"
                            chart_type = "bar chart"
                            
                            # Save chart data in session state
                            st.session_state.charts[chart_id] = {
                                "image": img_src,
                                "image_bytes": img_bytes,
                                "title": params["title"],
                                "data": df.to_dict(),
                                "chart_type": chart_type
                            }
                            
                            api_response = {
                                "success": True,
                                "message": f"Created bar chart titled '{params['title']}' with {len(df)} data points.",
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

                    # Create Line Chart function
                    if response.function_call.name == "create_line_chart":
                        job_config = bigquery.QueryJobConfig(
                            maximum_bytes_billed=100000000
                        )
                        try:
                            # Run the query to get data for the line chart
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
                            
                            # Create the line chart
                            fig, ax = plt.subplots(figsize=(12, 8))
                            
                            # Get the y columns (could be multiple for multi-line chart)
                            y_columns = params["y_columns"]
                            if isinstance(y_columns, str):
                                y_columns = [y_columns]
                                
                            # Plot each y column as a line
                            for y_col in y_columns:
                                ax.plot(df[params["x_column"]], df[y_col], marker='o', label=y_col)
                            
                            ax.set_xlabel(params["x_column"])
                            ax.set_ylabel(", ".join(y_columns))
                            ax.set_title(params["title"])
                            
                            # Add legend if there are multiple lines
                            if len(y_columns) > 1:
                                ax.legend()
                                
                            # Format x-axis if it looks like a date
                            if df[params["x_column"]].dtype == 'datetime64[ns]':
                                plt.gcf().autofmt_xdate()
                                
                            plt.grid(True, linestyle='--', alpha=0.7)
                            plt.tight_layout()
                            
                            # Save chart to memory for display
                            buf = io.BytesIO()
                            fig.savefig(buf, format='png', bbox_inches='tight')
                            buf.seek(0)
                            img_bytes = buf.getvalue()
                            img_b64 = base64.b64encode(img_bytes).decode()
                            img_src = f"data:image/png;base64,{img_b64}"
                            
                            # Generate a unique ID for this chart
                            chart_id = f"chart_{len(st.session_state.charts)}"
                            chart_type = "line chart"
                            
                            # Save chart data in session state
                            st.session_state.charts[chart_id] = {
                                "image": img_src,
                                "image_bytes": img_bytes,
                                "title": params["title"],
                                "data": df.to_dict(),
                                "chart_type": chart_type
                            }
                            
                            api_response = {
                                "success": True,
                                "message": f"Created line chart titled '{params['title']}' with {len(df)} data points.",
                                "chart_id": chart_id
                            }
                            api_response = str(api_response)
                            api_requests_and_responses.append(
                                [response.function_call.name, params, api_response]
                            )
                            
                        except Exception as e:
                            error_message = f"""
                            We're having trouble creating the line chart. This
                            could be due to an invalid query or unsuitable data structure.
                            Try rephrasing your question. Details:

                            {str(e)}"""
                            st.error(error_message)
                            api_response = error_message
                            api_requests_and_responses.append(
                                [response.function_call.name, params, api_response]
                            )

                    # Create Scatter Plot function
                    if response.function_call.name == "create_scatter_plot":
                        job_config = bigquery.QueryJobConfig(
                            maximum_bytes_billed=100000000
                        )
                        try:
                            # Run the query to get data for the scatter plot
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
                            
                            # Create the scatter plot
                            fig, ax = plt.subplots(figsize=(12, 8))
                            
                            # Check if a color column is provided
                            if "color_column" in params and params["color_column"]:
                                # Create a colorful scatter plot by category
                                categories = df[params["color_column"]].unique()
                                for category in categories:
                                    subset = df[df[params["color_column"]] == category]
                                    ax.scatter(
                                        subset[params["x_column"]], 
                                        subset[params["y_column"]], 
                                        label=category,
                                        alpha=0.7
                                    )
                                ax.legend(title=params["color_column"])
                            else:
                                # Create a simple scatter plot
                                ax.scatter(df[params["x_column"]], df[params["y_column"]], alpha=0.7)
                            
                            ax.set_xlabel(params["x_column"])
                            ax.set_ylabel(params["y_column"])
                            ax.set_title(params["title"])
                            
                            plt.grid(True, linestyle='--', alpha=0.5)
                            plt.tight_layout()
                            
                            # Save chart to memory for display
                            buf = io.BytesIO()
                            fig.savefig(buf, format='png', bbox_inches='tight')
                            buf.seek(0)
                            img_bytes = buf.getvalue()
                            img_b64 = base64.b64encode(img_bytes).decode()
                            img_src = f"data:image/png;base64,{img_b64}"
                            
                            # Generate a unique ID for this chart
                            chart_id = f"chart_{len(st.session_state.charts)}"
                            chart_type = "scatter plot"
                            
                            # Save chart data in session state
                            st.session_state.charts[chart_id] = {
                                "image": img_src,
                                "image_bytes": img_bytes,
                                "title": params["title"],
                                "data": df.to_dict(),
                                "chart_type": chart_type
                            }
                            
                            api_response = {
                                "success": True,
                                "message": f"Created scatter plot titled '{params['title']}' with {len(df)} data points.",
                                "chart_id": chart_id
                            }
                            api_response = str(api_response)
                            api_requests_and_responses.append(
                                [response.function_call.name, params, api_response]
                            )
                            
                        except Exception as e:
                            error_message = f"""
                            We're having trouble creating the scatter plot. This
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
                        "   - Function name: ```"
                        + str(api_requests_and_responses[-1][0])
                        + "```"
                    )
                    backend_details += "\n\n"
                    backend_details += (
                        "   - Function parameters: ```"
                        + str(api_requests_and_responses[-1][1])
                        + "```"
                    )
                    backend_details += "\n\n"
                    backend_details += (
                        "   - API response: ```"
                        + str(api_requests_and_responses[-1][2])
                        + "```"
                    )
                    backend_details += "\n\n"
                    with message_placeholder.container():
                        st.markdown(backend_details)

                except AttributeError:
                    function_calling_in_process = False

            time.sleep(3)

            full_response = response.text
            with message_placeholder.container():
                st.markdown(full_response.replace("$", r"\$"))  # noqa: W605
                
                # Display chart if one was created
                if chart_id and chart_id in st.session_state.charts:
                    chart_data = st.session_state.charts[chart_id]
                    st.image(chart_data["image"])
                    st.download_button(
                        label=f"Download {chart_
