# pylint: disable=broad-exception-caught,invalid-name

import time
import json
import ast
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from google import genai
from google.cloud import bigquery
from google.genai.types import FunctionDeclaration, GenerateContentConfig, Part, Tool

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
calculate_utilization_rate_func = FunctionDeclaration(
    name="calculate_utilization_rate",
    description="Calculate the Vehicle Utilization Rate as a percentage by dividing operated kilometers by planned kilometers",
    parameters={
        "type": "object",
        "properties": {
            "operated_km": {
                "type": "number",
                "description": "The actual kilometers operated by vehicles",
            },
            "planned_km": {
                "type": "number",
                "description": "The planned kilometers that vehicles should have operated",
            }
        },
        "required": [
            "operated_km",
            "planned_km",
        ],
    },
)


visualize_data_func = FunctionDeclaration(
    name="visualize_data",
    description="Generate a visualization of the data provided to answer the user's question",
    parameters={
        "type": "object",
        "properties": {
            "data": {
                "type": "string",
                "description": "JSON string of data to visualize or 'last_query_result' to use the last query results"
            },
            "chart_type": {
                "type": "string",
                "description": "Type of chart to generate (bar, line, pie, scatter, heatmap)",
                "enum": ["bar", "line", "pie", "scatter", "heatmap"]
            },
            "x_column": {
                "type": "string",
                "description": "Column to use for x-axis"
            },
            "y_column": {
                "type": "string",
                "description": "Column to use for y-axis (or value for pie charts)"
            },
            "title": {
                "type": "string",
                "description": "Title for the visualization"
            },
            "color_column": {
                "type": "string",
                "description": "Column to use for color grouping (optional)",
            }
        },
        "required": [
            "data",
            "chart_type",
            "x_column",
            "y_column",
            "title"
        ],
    },
)

sql_query_tool = Tool(
    function_declarations=[
        list_datasets_func,
        list_tables_func,
        get_table_func,
        sql_query_func,
        calculate_utilization_rate_func,
        visualize_data_func,
    ],
)

client = genai.Client(vertexai=True, location=LOCATION)

st.set_page_config(
    page_title="SQL Talk with BigQuery",
    page_icon="vertex-ai.png",
    layout="wide",
)

col1, col2 = st.columns([8, 1])
with col1:
    st.title("SQL Talk with BigQuery & Visualization")
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
        - What percentage of orders are returned? Show me a pie chart.
        - How is inventory distributed across our regional distribution centers? Visualize as a bar chart.
        - Do customers typically place more than one order? Show the distribution.
        - Which product categories have the highest profit margins? Create a visualization.
    """
    )

if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_query_data" not in st.session_state:
    st.session_state.last_query_data = None

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"].replace("$", r"\$"))  # noqa: W605
        try:
            if "visualization" in message and message["visualization"] is not None:
                st.plotly_chart(message["visualization"], use_container_width=True)
        except KeyError:
            pass
        try:
            with st.expander("Function calls, parameters, and responses"):
                st.markdown(message["backend_details"])
        except KeyError:
            pass

def generate_visualization(data_str, chart_type, x_column, y_column, title, color_column=None):
    try:
        # Convert string representation of data to actual list of dictionaries
        if isinstance(data_str, str):
            # First try standard JSON parsing
            try:
                data = json.loads(data_str)
            except json.JSONDecodeError:
                # If that fails, try to clean the string and use ast.literal_eval
                # Replace single quotes with double quotes for JSON compatibility
                cleaned_str = data_str.replace("'", '"')
                # Try to parse with json again
                try:
                    data = json.loads(cleaned_str)
                except json.JSONDecodeError:
                    # If still failing, use ast.literal_eval as a fallback
                    try:
                        data = ast.literal_eval(data_str)
                    except (SyntaxError, ValueError):
                        # If all parsing methods fail, return error
                        st.error(f"Could not parse data string: {data_str[:100]}...")
                        return None
        else:
            data = data_str
            
        # Convert to DataFrame
        df = pd.DataFrame(data)
        
        # Create the appropriate chart based on the type
        if chart_type == "bar":
            if color_column and color_column in df.columns:
                fig = px.bar(df, x=x_column, y=y_column, color=color_column, title=title)
            else:
                fig = px.bar(df, x=x_column, y=y_column, title=title)
                
        elif chart_type == "line":
            if color_column and color_column in df.columns:
                fig = px.line(df, x=x_column, y=y_column, color=color_column, title=title)
            else:
                fig = px.line(df, x=x_column, y=y_column, title=title)
                
        elif chart_type == "pie":
            fig = px.pie(df, names=x_column, values=y_column, title=title)
            
        elif chart_type == "scatter":
            if color_column and color_column in df.columns:
                fig = px.scatter(df, x=x_column, y=y_column, color=color_column, title=title)
            else:
                fig = px.scatter(df, x=x_column, y=y_column, title=title)
                
        elif chart_type == "heatmap":
            # Pivot the data for heatmap if necessary
            pivot_df = df.pivot(index=x_column, columns=color_column, values=y_column) if color_column else df
            fig = px.imshow(pivot_df, title=title)
        else:
            return None
            
        # Add styling
        fig.update_layout(
            template="plotly_white",
            title={
                'y':0.95,
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top'
            }
        )
        
        return fig
    except Exception as e:
        st.error(f"Error generating visualization: {str(e)}")
        return None

if prompt := st.chat_input("Ask me about information in the database..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        visualization = None
        
        chat = client.chats.create(
            model=MODEL_ID,
            config=GenerateContentConfig(temperature=0, tools=[sql_query_tool]),
        )
        client = bigquery.Client()

        enhanced_prompt = prompt + """
            I want the response to be complete and don't miss anything about what is asked. Only use information from BigQuery, and do not make up any data.
            If the user asks for visualization or chart, first query the data and then call the visualize_data function to generate the appropriate visualization.
            Choose the most appropriate chart type for the data and question being asked.
            """

        try:
            response = chat.send_message(enhanced_prompt)
            response = response.candidates[0].content.parts[0]

            print(response)

            api_requests_and_responses = []
            backend_details = ""
            query_result = None

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
                            query_result = [dict(row) for row in api_response]
                            st.session_state.last_query_data = query_result
                            api_response = str(query_result)
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
                    if response.function_call.name == "calculate_utilization_rate":
                        operated_km = float(params["operated_km"])
                        planned_km = float(params["planned_km"])
                        
                        # Avoid division by zero
                        if planned_km == 0:
                            utilization_rate = 0
                        else:
                            utilization_rate = (operated_km / planned_km) * 100
                        
                        api_response = f"{utilization_rate:.2f}%"
                        api_requests_and_responses.append(
                            [response.function_call.name, params, api_response]
                        )
                            
                    if response.function_call.name == "visualize_data":
                        try:
                            # Parse parameters
                            chart_data = params.get("data")
                            
                            # Handle the case where "last_query_result" is passed
                            if chart_data == "last_query_result" and st.session_state.last_query_data:
                                chart_data = st.session_state.last_query_data
                            
                            # If chart_data is a string and looks like a reference to last query
                            elif chart_data and isinstance(chart_data, str) and ("last" in chart_data.lower() or "previous" in chart_data.lower() or "query" in chart_data.lower()) and st.session_state.last_query_data:
                                chart_data = st.session_state.last_query_data
                            
                            chart_type = params.get("chart_type")
                            x_column = params.get("x_column")
                            y_column = params.get("y_column")
                            title = params.get("title")
                            color_column = params.get("color_column", None)
                            
                            # Optional debugging information
                            if st.session_state.get("debug_mode", False):
                                st.write(f"Debug - Data type: {type(chart_data)}")
                                if isinstance(chart_data, str):
                                    st.write(f"Debug - Data string (first 100 chars): {chart_data[:100]}...")
                            
                            # Generate the chart
                            visualization = generate_visualization(
                                chart_data, chart_type, x_column, y_column, title, color_column
                            )
                            
                            if visualization:
                                api_response = "Visualization successfully created."
                            else:
                                api_response = "Failed to create visualization."
                                
                            api_requests_and_responses.append(
                                [response.function_call.name, params, api_response]
                            )
                        except Exception as e:
                            error_message = f"Error generating visualization: {str(e)}"
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
                if visualization:
                    st.plotly_chart(visualization, use_container_width=True)
                with st.expander("Function calls, parameters, and responses:"):
                    st.markdown(backend_details)

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": full_response,
                    "backend_details": backend_details,
                    "visualization": visualization
                }
            )
        except Exception as e:
            print(e)
            error_message = f"""
                Something went wrong! We encountered an unexpected error while
                trying to process your request. Please try rephrasing your
                question. Details:

                {str(e)}"""
            st.error(error_message)
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": error_message,
                }
            )

# Add a debug mode toggle in the sidebar
with st.sidebar:
    st.title("Settings")
    debug_mode = st.checkbox("Enable Debug Mode", value=False)
    if debug_mode:
        st.session_state["debug_mode"] = True
    else:
        st.session_state["debug_mode"] = False
    
    if st.button("Clear Conversation"):
        st.session_state.messages = []
        st.session_state.last_query_data = None
        st.experimental_rerun()
