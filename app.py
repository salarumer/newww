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

st.markdown(
    """
    <style>
    /* Set background color */
    html, body, [class*="st-"] {
        background-color: white !important;
        color: black !important;
    }

    /* Update text input fields */
    input, textarea {
        background-color: white !important;
        color: black !important;
        border: 1px solid #ccc !important;
    }

    /* Update buttons */
    button {
        background-color: #f0f0f0 !important;
        color: black !important;
        border: 1px solid #bbb !important;
    }

    /* Update select dropdowns */
    select {
        background-color: white !important;
        color: black !important;
    }

    /* Update Streamlit widgets */
    .stTextInput, .stTextArea, .stSelectbox, .stButton {
        background-color: white !important;
        color: black !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

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

calculate_all_punctuality_func = FunctionDeclaration(
    name="calculate_all_punctuality",
    description="Calculate the performance rates for all punctuality categories (Ontime, Early, Late, Pending) as percentages based on punctuality data",
    parameters={
        "type": "object",
        "properties": {
            "punctuality_data": {
                "type": "string",
                "description": "JSON string containing data with punctuality categories. Format: [{\"punctuality_category\": \"Ontime\", \"count\": 120}, ...]"
            }
        },
        "required": [
            "punctuality_data",
        ],
    },
)

calculate_seat_occupancy_func = FunctionDeclaration(
    name="calculate_seat_occupancy",
    description="Calculate the Seat Occupancy Rate as a percentage based on total passengers and available seats.",
    parameters={
        "type": "object",
        "properties": {
            "total_passengers": {
                "type": "integer",
                "description": "Total number of passengers on the vehicle."
            },
            "total_available_seats": {
                "type": "integer",
                "description": "Total number of available seats in the vehicle."
            }
        },
        "required": [
            "total_passengers",
            "total_available_seats",
        ],
    },
)


calculate_performance_metrics_func = FunctionDeclaration(
    name="calculate_performance_metrics",
    description="Calculate utilization rate, seat occupancy, and all punctuality metrics (Ontime, Early, Late, Pending) together",
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
            },
            "punctuality_data": {
                "type": "string",
                "description": "JSON string containing data with punctuality categories. Format: [{\"punctuality_category\": \"Ontime\", \"count\": 120}, ...]"
            },
            "total_passengers": {
                "type": "integer",
                "description": "Total number of passengers on the vehicle (optional)."
            },
            "total_available_seats": {
                "type": "integer",
                "description": "Total number of available seats in the vehicle (optional)."
            }
        },
        "required": [
            "operated_km",
            "planned_km",
            "punctuality_data",
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
        calculate_all_punctuality_func,
        calculate_seat_occupancy_func,
        calculate_performance_metrics_func,
        visualize_data_func,
    ],
)

client = genai.Client(vertexai=True, location=LOCATION)

if "messages" not in st.session_state:
    st.session_state.messages = []




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
            
        # Create DataFrame
        df = pd.DataFrame(data)
        
        # Log the columns for debugging
        print(f"Original columns: {df.columns.tolist()}")
        
        # Check if we need to preprocess data for specific routes
        route_pattern = re.compile(r'route_(\w+)_(\w+)')
        route_specific_columns = [col for col in df.columns if route_pattern.match(col)]
        
        # Handle specific route metrics (both performance and ridership)
        if route_specific_columns:
            # Extract route information from column names
            route_data = []
            for col in route_specific_columns:
                match = route_pattern.match(col)
                if match:
                    route_id = match.group(1)
                    metric = match.group(2)
                    for idx, value in enumerate(df[col]):
                        # Create a row for each route-metric combination
                        row_data = {'route': route_id, 'metric': metric, 'value': value}
                        # Add any other columns from the original dataframe
                        for other_col in df.columns:
                            if other_col not in route_specific_columns:
                                row_data[other_col] = df.iloc[idx][other_col]
                        route_data.append(row_data)
            
            # Create a new dataframe with the restructured data
            if route_data:
                df = pd.DataFrame(route_data)
        
        # Handle single route performance metrics
        elif any(col.startswith('route_') for col in df.columns) or 'route' in df.columns:
            # Identify the route column
            route_col = next((col for col in df.columns if col.startswith('route_')), 'route')
            
            # Check if we have performance metrics that should be visualized
            performance_metrics = [
                col for col in df.columns 
                if any(metric in col.lower() for metric in ['utilization', 'punctuality', 'occupancy', 'ontime', 'late', 'early', 'pending'])
            ]
            
            # If we have performance metrics, restructure for visualization
            if performance_metrics:
                metrics_data = []
                for idx, row in df.iterrows():
                    route_id = row[route_col]
                    for metric in performance_metrics:
                        metrics_data.append({
                            'route': route_id,
                            'metric': metric,
                            'value': row[metric]
                        })
                if metrics_data:
                    df = pd.DataFrame(metrics_data)
        
        # If we're dealing with a specific route query, filter for that route
        route_mentions = re.findall(r'route[_\s]*(\w+)', title.lower() + ' ' + x_column.lower() + ' ' + y_column.lower())
        if route_mentions and 'route' in df.columns:
            mentioned_routes = set(route_mentions)
            df = df[df['route'].astype(str).str.lower().isin([r.lower() for r in mentioned_routes])]
        
        # Log the processed dataframe
        print(f"Processed dataframe columns: {df.columns.tolist()}")
        print(f"Processed dataframe shape: {df.shape}")
        
        # Determine the best columns for visualization
        columns = df.columns.tolist()
        
        def find_best_match(target, columns):
            # Exact match
            if target in columns:
                return target
            
            # Check if target is contained within any column name
            contained_matches = [col for col in columns if target.lower() in col.lower()]
            if contained_matches:
                return contained_matches[0]
            
            # Check if any column contains the target
            container_matches = [col for col in columns if col.lower() in target.lower()]
            if container_matches:
                return container_matches[0]
            
            # For 'route', try to find any column that might contain route information
            if target.lower() == 'route':
                route_cols = [col for col in columns if 'route' in col.lower()]
                if route_cols:
                    return route_cols[0]
            
            # Return the first column as fallback for critical axes
            if columns:
                return columns[0]
            
            return None
        
        # Special case for performance metrics visualization
        if 'metric' in columns and 'value' in columns:
            actual_x = 'metric'
            actual_y = 'value'
            actual_color = 'route' if 'route' in columns else color_column
        else:
            # Find the best matching columns
            actual_x = find_best_match(x_column, columns)
            actual_y = find_best_match(y_column, columns)
            actual_color = find_best_match(color_column, columns) if color_column else None
        
        # Special case for route performance comparison
        if 'route' in columns and any(metric in columns for metric in ['value', 'riders', 'count']):
            if chart_type == "bar":
                value_col = next((col for col in ['value', 'riders', 'count'] if col in columns), columns[-1])
                if 'metric' in columns:
                    # Create a grouped bar chart for multiple metrics
                    fig = px.bar(df, x='route', y=value_col, color='metric', barmode='group', title=title)
                else:
                    # Create a simple bar chart for one metric
                    fig = px.bar(df, x='route', y=value_col, title=title)
                return fig
        
        # If we still don't have valid columns, log an error
        if not actual_x or not actual_y:
            error_msg = f"Could not find appropriate columns. Looking for '{x_column}' and '{y_column}' but found {columns}"
            st.error(error_msg)
            return None
            
        # Create the appropriate chart based on the type
        if chart_type == "bar":
            if actual_color and actual_color in df.columns:
                fig = px.bar(df, x=actual_x, y=actual_y, color=actual_color, title=title)
            else:
                fig = px.bar(df, x=actual_x, y=actual_y, title=title)
                
        elif chart_type == "line":
            if actual_color and actual_color in df.columns:
                fig = px.line(df, x=actual_x, y=actual_y, color=actual_color, title=title)
            else:
                fig = px.line(df, x=actual_x, y=actual_y, title=title)
                
        elif chart_type == "pie":
            fig = px.pie(df, names=actual_x, values=actual_y, title=title)
            
        elif chart_type == "scatter":
            if actual_color and actual_color in df.columns:
                fig = px.scatter(df, x=actual_x, y=actual_y, color=actual_color, title=title)
            else:
                fig = px.scatter(df, x=actual_x, y=actual_y, title=title)
                
        elif chart_type == "heatmap":
            # Pivot the data for heatmap if necessary
            if actual_color and actual_color in df.columns:
                pivot_df = df.pivot(index=actual_x, columns=actual_color, values=actual_y)
            else:
                # For heatmap without a color column, try to detect a sensible pivoting strategy
                numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
                if len(numeric_cols) > 0 and len(df.columns) > 2:
                    # Use the first non-x column that's not y as the pivot column
                    potential_pivot_cols = [col for col in df.columns if col != actual_x and col != actual_y]
                    if potential_pivot_cols:
                        pivot_df = df.pivot(index=actual_x, columns=potential_pivot_cols[0], values=actual_y)
                    else:
                        pivot_df = df
                else:
                    pivot_df = df
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
        # Print full traceback for debugging
        import traceback
        print(traceback.format_exc())
        return None

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
            
            When asked about performance metrics, calculate utilization rate, 
            seat occupancy, and all punctuality categories including Ontime, Early, 
            Late, and Pending. Always provide a breakdown of all punctuality categories.

            IMPORTANT: The punctuality_category column has exactly four values: 
            'Ontime', 'Late', 'Pending', and 'Early'. Include all categories in 
            your analysis.

            If the user asks for visualization or chart, first query the data and then call the visualize_data function to generate the appropriate visualization.
            Choose the most appropriate chart type for the data and question being asked
            """

        try:
            response = chat.send_message(prompt)
            response = response.candidates[0].content.parts[0]

            print(response)

            api_requests_and_responses = []
            backend_details = ""
            
            # Initialize visualization as None
            visualization = None

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
                        table = client.get_table(params["table_id"])
                        api_repr = table.to_api_repr()
                        
                        # Extract the description and column names safely
                        description = api_repr.get("description", "")
                        
                        # Make sure schema exists and has fields before trying to access
                        column_names = []
                        if "schema" in api_repr and "fields" in api_repr["schema"]:
                            column_names = [col.get("name", "") for col in api_repr["schema"]["fields"]]
                        
                        # Store the extracted information
                        extracted_info = [
                            str(description),
                            str(column_names)
                        ]
                        
                        api_requests_and_responses.append(
                            [
                                response.function_call.name,
                                params,
                                extracted_info,
                            ]
                        )
                        
                        # Convert the full response to string for the model
                        api_response = str(api_repr)

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
                            # Store last query result for visualization
                            st.session_state.last_query_data = api_response
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
                    if response.function_call.name == "calculate_seat_occupancy":
                        total_passengers = int(params["total_passengers"])
                        total_available_seats = int(params["total_available_seats"])
                        
                        # Avoid division by zero
                        if total_available_seats == 0:
                            seat_occupancy_rate = 0
                        else:
                            seat_occupancy_rate = (total_passengers / total_available_seats) * 100
                        
                        api_response = f"{seat_occupancy_rate:.2f}%"
                        api_requests_and_responses.append(
                            [response.function_call.name, params, api_response]
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
                    
                    if response.function_call.name == "calculate_all_punctuality":
                        try:
                            # Parse the punctuality data
                            punctuality_data = json.loads(params["punctuality_data"])
                            
                            # Calculate total trips
                            total_trips = sum(item["count"] for item in punctuality_data)
                            
                            # Initialize counts dictionary for all categories
                            categories = {
                                "Ontime": 0,
                                "Early": 0, 
                                "Late": 0,
                                "Pending": 0
                            }
                            
                            # Populate counts for categories that exist in the data
                            for item in punctuality_data:
                                category = item["punctuality_category"]
                                if category in categories:
                                    categories[category] = item["count"]
                            
                            # Calculate percentages for all categories
                            punctuality_metrics = {}
                            for category, count in categories.items():
                                if total_trips == 0:
                                    percentage = 0
                                else:
                                    percentage = (count / total_trips) * 100
                                punctuality_metrics[f"{category.lower()}_trips"] = count
                                punctuality_metrics[f"{category.lower()}_percentage"] = f"{percentage:.2f}%"
                            
                            punctuality_metrics["total_trips"] = total_trips
                            
                            api_response = json.dumps(punctuality_metrics)
                            api_requests_and_responses.append(
                                [response.function_call.name, params, api_response]
                            )
                        except Exception as e:
                            error_message = f"Error calculating punctuality metrics: {str(e)}"
                            st.error(error_message)
                            api_response = error_message
                            api_requests_and_responses.append(
                                [response.function_call.name, params, api_response]
                            )
                    
                    if response.function_call.name == "calculate_performance_metrics":
                        try:
                            # Calculate utilization rate
                            operated_km = float(params["operated_km"])
                            planned_km = float(params["planned_km"])
                            
                            # Avoid division by zero
                            if planned_km == 0:
                                utilization_rate = 0
                            else:
                                utilization_rate = (operated_km / planned_km) * 100
                            
                            # Parse the punctuality data
                            punctuality_data = json.loads(params["punctuality_data"])
                            
                            # Calculate total trips
                            total_trips = sum(item["count"] for item in punctuality_data)
                            
                            # Initialize counts dictionary for all categories
                            categories = {
                                "Ontime": 0,
                                "Early": 0, 
                                "Late": 0,
                                "Pending": 0
                            }
                            
                            # Populate counts for categories that exist in the data
                            for item in punctuality_data:
                                category = item["punctuality_category"]
                                if category in categories:
                                    categories[category] = item["count"]
                            
                            # Calculate seat occupancy if provided
                            seat_occupancy_rate = None
                            if "total_passengers" in params and "total_available_seats" in params:
                                total_passengers = int(params["total_passengers"])
                                total_available_seats = int(params["total_available_seats"])
                                
                                if total_available_seats == 0:
                                    seat_occupancy_rate = 0
                                else:
                                    seat_occupancy_rate = (total_passengers / total_available_seats) * 100
                            
                            # Prepare the comprehensive metrics response
                            performance_metrics = {
                                "utilization_rate": f"{utilization_rate:.2f}%",
                                "operated_km": operated_km,
                                "planned_km": planned_km,
                                "total_trips": total_trips
                            }
                            
                            # Add seat occupancy if calculated
                            if seat_occupancy_rate is not None:
                                performance_metrics["seat_occupancy_rate"] = f"{seat_occupancy_rate:.2f}%"
                                performance_metrics["total_passengers"] = total_passengers
                                performance_metrics["total_available_seats"] = total_available_seats
                            
                            # Add punctuality metrics for all categories
                            for category, count in categories.items():
                                if total_trips == 0:
                                    percentage = 0
                                else:
                                    percentage = (count / total_trips) * 100
                                performance_metrics[f"{category.lower()}_trips"] = count
                                performance_metrics[f"{category.lower()}_percentage"] = f"{percentage:.2f}%"
                            
                            api_response = json.dumps(performance_metrics)
                            api_requests_and_responses.append(
                                [response.function_call.name, params, api_response]
                            )
                        except Exception as e:
                            error_message = f"Error calculating performance metrics: {str(e)}"
                            st.error(error_message)
                            api_response = error_message
                            api_requests_and_responses.append(
                                [response.function_call.name, params, api_response]
                            )
                    if response.function_call.name == "visualize_data":
                        try:
                            # Parse parameters
                            chart_data = params.get("data")
                            
                            # Handle the case where "last_query_result" is passed
                            if chart_data == "last_query_result" and hasattr(st.session_state, 'last_query_data'):
                                chart_data = st.session_state.last_query_data
                            
                            # If chart_data is a string and looks like a reference to last query
                            elif chart_data and isinstance(chart_data, str) and ("last" in chart_data.lower() or "previous" in chart_data.lower() or "query" in chart_data.lower()) and hasattr(st.session_state, 'last_query_data'):
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
                            visualization = None  # Ensure visualization is set to None on error
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
