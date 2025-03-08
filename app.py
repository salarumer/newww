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

sql_query_tool = Tool(
    function_declarations=[
        list_datasets_func,
        list_tables_func,
        get_table_func,
        sql_query_func,
        create_pie_chart_func,
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
    body {
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

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"].replace("$", r"\$"))  # noqa: W605
        
        # Display pie chart if available
        if "chart_id" in message and message["chart_id"] in st.session_state.pie_charts:
            chart_data = st.session_state.pie_charts[message["chart_id"]]
            st.image(chart_data["image"])
            st.download_button(
                label="Download Pie Chart",
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
            from BigQuery, do not make up information. If the user's query mentions a visualization, 
            chart, or specifically a pie chart, use the create_pie_chart function to generate a pie chart 
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
                            chart_id = f"chart_{len(st.session_state.pie_charts)}"
                            
                            # Save chart data in session state
                            st.session_state.pie_charts[chart_id] = {
                                "image": img_src,
                                "image_bytes": img_bytes,
                                "title": params["title"],
                                "data": df.to_dict()
                            }
                            
                            # Display the chart
                            chart_placeholder = st.empty()
                            chart_placeholder.image(img_src)
                            
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
                
                # Display pie chart if one was created
                if chart_id and chart_id in st.session_state.pie_charts:
                    chart_data = st.session_state.pie_charts[chart_id]
                    st.image(chart_data["image"])
                    st.download_button(
                        label="Download Pie Chart",
                        data=chart_data["image_bytes"],
                        file_name=f"{chart_data['title'].replace(' ', '_')}.png",
                        mime="image/png"
                    )
                
                with st.expander("Function calls, parameters, and responses:"):
                    st.markdown(backend_details)

            message_data = {
                "role": "assistant",
                "content": full_response,
                "backend_details": backend_details,
            }
            
            # Add chart ID to message if one was created
            if chart_id:
                message_data["chart_id"] = chart_id
                
            st.session_state.messages.append(message_data)
            
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
