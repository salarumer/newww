# pylint: disable=broad-exception-caught,invalid-name
import time
import json
import ast
import pandas as pd
from datetime import datetime

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
COST_PER_KM = 2.4  # Fixed cost per km in AED

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
service_reliability_func = FunctionDeclaration(
    name="calculate_service_reliability",
    description="Calculate the Service Reliability as a percentage based on operated kilometers and lost kilometers.",
    parameters={
        "type": "object",
        "properties": {
            "operated_km": {
                "type": "number",
                "description": "Total kilometers actually operated by the vehicle."
            },
            "lost_km": {
                "type": "number",
                "description": "Total kilometers lost or not operated as planned."
            }
        },
        "required": [
            "operated_km",
            "lost_km",
        ],
    },
)

lost_km_rate_func = FunctionDeclaration(
    name="calculate_lost_km_rate",
    description="Calculate the Lost Kilometer Rate as a percentage based on lost kilometers and planned kilometers.",
    parameters={
        "type": "object",
        "properties": {
            "lost_km": {
                "type": "number",
                "description": "Total kilometers lost or not operated as planned."
            },
            "planned_km": {
                "type": "number",
                "description": "Total kilometers planned to be operated."
            }
        },
        "required": [
            "lost_km",
            "planned_km",
        ],
    },
)
service_utilization_rate_func = FunctionDeclaration(
    name="calculate_service_utilization_rate",
    description="Calculate the Service Utilization Rate as a percentage based on operated seat-kilometers and planned seat-kilometers.",
    parameters={
        "type": "object",
        "properties": {
            "operated_seats_km": {
                "type": "number",
                "description": "Total seat-kilometers operated (seats × kilometers)."
            },
            "planned_seats_km": {
                "type": "number",
                "description": "Total seat-kilometers planned (seats × kilometers)."
            }
        },
        "required": [
            "operated_seats_km",
            "planned_seats_km",
        ],
    },
)
route_efficiency_func = FunctionDeclaration(
    name="calculate_route_efficiency",
    description="Calculate the Route Efficiency Ratio as a percentage based on operated kilometers and planned kilometers.",
    parameters={
        "type": "object",
        "properties": {
            "operated_km": {
                "type": "number",
                "description": "Total kilometers actually operated by the vehicles."
            },
            "planned_km": {
                "type": "number",
                "description": "Total kilometers planned to be operated."
            }
        },
        "required": [
            "operated_km",
            "planned_km",
        ],
    },
)
breakdown_rate_func = FunctionDeclaration(
    name="calculate_breakdown_rate",
    description="Calculate the Breakdown Rate (Mechanical Reliability) as a percentage based on kilometers lost due to breakdowns and total operated kilometers.",
    parameters={
        "type": "object",
        "properties": {
            "lost_km_out_of_control": {
                "type": "number",
                "description": "Kilometers lost due to mechanical breakdowns or issues outside of control."
            },
            "operated_km": {
                "type": "number",
                "description": "Total kilometers actually operated by the vehicles."
            }
        },
        "required": [
            "lost_km_out_of_control",
            "operated_km",
        ],
    },
)
trip_completion_rate_func = FunctionDeclaration(
    name="calculate_trip_completion_rate",
    description="Calculate the Trip Completion Rate as a percentage based on completed trips and total planned trips.",
    parameters={
        "type": "object",
        "properties": {
            "trips_completed": {
                "type": "integer",
                "description": "Number of trips that were successfully completed."
            },
            "total_planned_trips": {
                "type": "integer",
                "description": "Total number of trips that were planned."
            }
        },
        "required": [
            "trips_completed",
            "total_planned_trips",
        ],
    },
)

calculate_farebox_recovery_ratio_func = FunctionDeclaration(
    name="calculate_farebox_recovery_ratio",
    description="Calculate the Farebox Recovery Ratio as a percentage based on fare revenue and operating costs.",
    parameters={
        "type": "object",
        "properties": {
            "fare_by_card_passengers": {
                "type": "number",
                "description": "Total fare revenue from passengers paying by card."
            },
            "fare_by_cash_passengers": {
                "type": "number",
                "description": "Total fare revenue from passengers paying by cash."
            },
            "operated_km": {
                "type": "number",
                "description": "Total kilometers operated."
            }
        },
        "required": [
            "fare_by_card_passengers",
            "fare_by_cash_passengers",
            "operated_km",
        ],
    },
)
service_consistency_func = FunctionDeclaration(
    name="calculate_service_consistency",
    description="Calculate the Service Consistency as the standard deviation of punctuality differences.",
    parameters={
        "type": "object",
        "properties": {
            "punctuality_differences": {
                "type": "array",
                "items": {"type": "number"},
                "description": "List of punctuality differences (in minutes or seconds) for each trip."
            }
        },
        "required": [
            "punctuality_differences",
        ],
    },
)
get_performance_metrics_func = FunctionDeclaration(
    name="get_performance_metrics",
    description="Get a list of all available transit performance metrics and their descriptions.",
    parameters={
        "type": "object",
        "properties": {},
        "required": [],
    },

)

check_date_availability_func = FunctionDeclaration(
    name="check_date_availability",
    description="Check if data for a specific month is available in the tansit table from  dataset1",
    parameters={
        "type": "object",
        "properties": {
            "month": {
                "type": "string",
                "description": "Month to check in YYYY-MM format (e.g., '2025-03' for March 2025)",
            }
        },
        "required": [
            "month",
        ],
    },
)

monthly_summary_func = FunctionDeclaration(
    name="generate_monthly_summary",
    description="Generate a summary of transit performance metrics for the specified month",
    parameters={
        "type": "object",
        "properties": {
            "month": {
                "type": "string",
                "description": "Month to generate summary for in YYYY-MM format (e.g., '2025-03' for March 2025)"
            }
        },
        "required": [
            "month",
        ],
    },
)
sql_query_tool = Tool(
    function_declarations=[
        list_datasets_func,
        list_tables_func,
        get_table_func,
        sql_query_func,
        calculate_seat_occupancy_func,
         service_reliability_func,
        lost_km_rate_func,
        service_utilization_rate_func,
        route_efficiency_func,
        breakdown_rate_func,
        trip_completion_rate_func,
        calculate_farebox_recovery_ratio_func,
        get_performance_metrics_func,
        service_consistency_func,
        check_date_availability_func,
        monthly_summary_func,
    ],
)

client = genai.Client(vertexai=True, location=LOCATION)
if "messages" not in st.session_state:
    st.session_state.messages = []
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"].replace("$", r"\$"))  # noqa: W605
        
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
        bq_client = bigquery.Client()
        
        # Check if the user is asking for a monthly summary
        if "monthly summary" in prompt.lower() or "this month" in prompt.lower():
            # Add context about the current month to help model understand what's needed
            current_month = datetime.now().strftime("%Y-%m")  # Format: YYYY-MM
            month_name = datetime.now().strftime("%B %Y")  # Format: March 2025
            
            prompt += f"""
            The current month is {month_name} ({current_month}). First check if data for this month 
            is available using the check_date_availability function. If data is available, use the 
            generate_monthly_summary function to provide performance metrics for this month.
            """
        
        prompt += """
            Please give a concise, high-level summary followed by detail in
            plain language about where the information in your response is
            coming from in the database. Only use information that you learn
            from BigQuery, do not make up information.
        
            """
        try:
            response = chat.send_message(prompt)
            response = response.candidates[0].content.parts[0]
            print(response)
            api_requests_and_responses = []
            backend_details = ""
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
                        api_response = bq_client.list_datasets()
                        api_response = BIGQUERY_DATASET_ID
                        api_requests_and_responses.append(
                            [response.function_call.name, params, api_response]
                        )
                    if response.function_call.name == "list_tables":
                        api_response = bq_client.list_tables(params["dataset_id"])
                        api_response = str([table.table_id for table in api_response])
                        api_requests_and_responses.append(
                            [response.function_call.name, params, api_response]
                        )
                    if response.function_call.name == "get_table":
                        table = bq_client.get_table(params["table_id"])
                        api_repr = table.to_api_repr()
                        description = api_repr.get("description", "")
                        column_names = []
                        if "schema" in api_repr and "fields" in api_repr["schema"]:
                            column_names = [col.get("name", "") for col in api_repr["schema"]["fields"]]
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
                            query_job = bq_client.query(
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
                                }   )
                    if response.function_call.name == "calculate_seat_occupancy":
                        total_passengers = int(params["total_passengers"])
                        total_available_seats = int(params["total_available_seats"])
                        if total_available_seats == 0:
                            seat_occupancy_rate = 0
                        else:
                            seat_occupancy_rate = (total_passengers / total_available_seats) * 100
                        
                        api_response = f"{seat_occupancy_rate:.2f}%"
                        api_requests_and_responses.append(
                            [response.function_call.name, params, api_response]
                        )
                    if response.function_call.name == "calculate_service_reliability":
                        operated_km = float(params["operated_km"])
                        lost_km = float(params["lost_km"])
                        
                        total_planned_km = operated_km + lost_km
                        
                        if total_planned_km == 0:
                            service_reliability = 0
                        else:
                            service_reliability = (operated_km / total_planned_km) * 100
                        
                        api_response = f"{service_reliability:.2f}%"
                        api_requests_and_responses.append(
                            [response.function_call.name, params, api_response]
                        )
                  
                    if response.function_call.name == "calculate_lost_km_rate":
                        lost_km = float(params["lost_km"])
                        planned_km = float(params["planned_km"])
                        
                        if planned_km == 0:
                            lost_km_rate = 0
                        else:
                            lost_km_rate = (lost_km / planned_km) * 100
                        
                        api_response = f"{lost_km_rate:.2f}%"
                        api_requests_and_responses.append(
                            [response.function_call.name, params, api_response]
                        )
                    if response.function_call.name == "calculate_farebox_recovery_ratio":
                        fare_by_card_passengers = float(params["fare_by_card_passengers"])
                        fare_by_cash_passengers = float(params["fare_by_cash_passengers"])
                        operated_km = float(params["operated_km"])
                        
                        # Using static value for cost_per_km instead of getting it from params
                        cost_per_km = 2.0
                        
                        total_fare_revenue = fare_by_card_passengers + fare_by_cash_passengers
                        estimated_operating_cost = operated_km * cost_per_km
                        
                        if estimated_operating_cost == 0:
                            farebox_recovery_ratio = 0
                        else:
                            farebox_recovery_ratio = (total_fare_revenue / estimated_operating_cost) * 100
                        
                        api_response = f"{farebox_recovery_ratio:.2f}%"
                        api_requests_and_responses.append(
                            [response.function_call.name, params, api_response]
                        )
                    if response.function_call.name == "calculate_service_utilization_rate":
                        operated_seats_km = float(params["operated_seats_km"])
                        planned_seats_km = float(params["planned_seats_km"])
                        
                        if planned_seats_km == 0:
                            utilization_rate = 0
                        else:
                            utilization_rate = (operated_seats_km / planned_seats_km) * 100
                        
                        api_response = f"{utilization_rate:.2f}%"
                        api_requests_and_responses.append(
                            [response.function_call.name, params, api_response]
                        )
                    if response.function_call.name == "calculate_route_efficiency":
                        operated_km = float(params["operated_km"])
                        planned_km = float(params["planned_km"])
                        
                        if planned_km == 0:
                            route_efficiency = 0
                        else:
                            route_efficiency = (operated_km / planned_km) * 100
                        
                        api_response = f"{route_efficiency:.2f}%"
                        api_requests_and_responses.append(
                            [response.function_call.name, params, api_response]
                        )
                    if response.function_call.name == "calculate_breakdown_rate":
                        lost_km_out_of_control = float(params["lost_km_out_of_control"])
                        operated_km = float(params["operated_km"])
                        
                        if operated_km == 0:
                            breakdown_rate = 0
                        else:
                            breakdown_rate = (lost_km_out_of_control / operated_km) * 100
                        
                        api_response = f"{breakdown_rate:.2f}%"
                        api_requests_and_responses.append(
                            [response.function_call.name, params, api_response]
                        )
                    if response.function_call.name == "calculate_trip_completion_rate":
                        trips_completed = int(params["trips_completed"])
                        total_planned_trips = int(params["total_planned_trips"])
                        
                        if total_planned_trips == 0:
                            trip_completion_rate = 0
                        else:
                            trip_completion_rate = (trips_completed / total_planned_trips) * 100
                        
                        api_response = f"{trip_completion_rate:.2f}%"
                        api_requests_and_responses.append(
                            [response.function_call.name, params, api_response]
                        )
                    if response.function_call.name == "calculate_service_consistency":
   
    
                        punctuality_differences = params["punctuality_differences"]
                        
                        if not punctuality_differences or len(punctuality_differences) <= 1:
                            service_consistency = 0
                        else:
                            service_consistency = np.std(punctuality_differences)
                        
                        api_response = f"{service_consistency:.2f} minutes"  # Assuming differences are in minutes
                        api_requests_and_responses.append(
                            [response.function_call.name, params, api_response]
                        )   
                    if response.function_call.name == "get_performance_metrics":
                         performance_metrics = [
                     {
            "metric_name": "Seat Occupancy Rate",
            "description": "Percentage of available seats occupied by passengers.",
            "formula": "(Total Passengers / Total Available Seats) × 100",
            "example": "75% seat occupancy means 75% of available seats are filled."
                    },
                    {
            "metric_name": "Service Reliability",
            "description": "Percentage of planned service that was actually operated.",
            "formula": "(Operated KM / (Operated KM + Lost KM)) × 100",
            "example": "98% service reliability means 98% of planned service was completed."
                 },
                 {
            "metric_name": "Lost Kilometer Rate",
            "description": "Percentage of kilometers lost due to breakdowns, delays, or disruptions.",
            "formula": "(Lost KM / Planned KM) × 100",
            "example": "5% lost km rate means 5 out of 100 planned kilometers were not completed."
                 },
                 {
            "metric_name": "Service Utilization Rate",
            "description": "Percentage of planned seat kilometers that were actually used.",
            "formula": "(Operated Seats × KM / Planned Seats × KM) × 100",
            "example": "80% utilization means 80% of planned seat capacity was used."
                 },
                 {
            "metric_name": "Route Efficiency Ratio",
            "description": "Compares actual distance traveled vs. planned distance.",
            "formula": "(Operated KM / Planned KM) × 100",
            "example": "95% route efficiency means vehicles are mostly following planned routes."
                  },
                 {
            "metric_name": "Breakdown Rate",
            "description": "Measures how often vehicles experience issues per kilometer traveled.",
            "formula": "(Lost KM Out of Control / Operated KM) × 100",
            "example": "2% breakdown rate means 2 out of 100 km are lost due to breakdowns."
                 },
                 {
            "metric_name": "Trip Completion Rate",
            "description": "Percentage of scheduled trips that are completed.",
            "formula": "(Trips Completed / Total Planned Trips) × 100",
            "example": "98% trip completion rate means 2% of scheduled trips were missed."
                  },
                 {
            "metric_name": "Farebox Recovery Ratio",
            "description": "Percentage of operating expenses covered by fare revenue.",
            "formula": "(Total Fare Revenue / Operating Expenses) × 100",
            "example": "60% farebox recovery means 60% of operating costs are covered by fares."
                 }
                 ]                        
                         api_response = {"available_metrics": performance_metrics}
                         api_requests_and_responses.append(
                            [response.function_call.name, params, api_response]
                        )
                    if response.function_call.name == "check_date_availability":
                        month = params["month"]
                        # Query to check if data exists for the specified month
                        date_query = f"""
                        SELECT COUNT(*) as record_count
                        FROM {BIGQUERY_DATASET_ID}.tansittable
                        WHERE FORMAT_DATE('%Y-%m', service_day) = '{month}'
                        """
                        
                        job_config = bigquery.QueryJobConfig(maximum_bytes_billed=100000000)
                        query_job = bq_client.query(date_query, job_config=job_config)
                        date_result = query_job.result()
                        record_count = list(date_result)[0].record_count
                        
                        if record_count > 0:
                            api_response = json.dumps({
                                "available": True,
                                "month": month,
                                "record_count": record_count
                            })
                        else:
                            api_response = json.dumps({
                                "available": False,
                                "month": month,
                                "record_count": 0
                            })
                        
                        api_requests_and_responses.append(
                            [response.function_call.name, params, api_response]
                        )
                    if response.function_call.name == "generate_monthly_summary":
                        month = params["month"]
                        
                        # Query BigQuery for seat occupancy data using your actual column names
                        occupancy_query = f"""
                        SELECT 
                            AVG(total_passengers_on_vehicle) as avg_passengers,
                            AVG(actual_number_of_seats_in_vehicle) as avg_seats
                        FROM 
                            {BIGQUERY_DATASET_ID}.tansittable
                        WHERE 
                            FORMAT_DATE('%Y-%m', service_day) = '{month}'
                        """
                        
                        job_config = bigquery.QueryJobConfig(maximum_bytes_billed=100000000)
                        query_job = bq_client.query(occupancy_query, job_config=job_config)
                        occupancy_result = query_job.result()
                        occupancy_data = list(occupancy_result)[0]
                        
                        # Calculate seat occupancy
                        avg_passengers = occupancy_data.avg_passengers or 0
                        avg_seats = occupancy_data.avg_seats or 0
                        if avg_seats == 0:
                            seat_occupancy = 0
                        else:
                            seat_occupancy = (avg_passengers / avg_seats) * 100
                        
                        # Query BigQuery for lost km data
                        km_query = f"""
                        SELECT 
                            SUM(lost_km) as total_lost_km,
                            SUM(planned_km) as total_planned_km
                        FROM 
                            {BIGQUERY_DATASET_ID}.tansittable
                        WHERE 
                            FORMAT_DATE('%Y-%m', service_day) = '{month}'
                        """
                        
                        query_job = bq_client.query(km_query, job_config=job_config)
                        km_result = query_job.result()
                        km_data = list(km_result)[0]
                        
                        # Calculate lost km rate
                        total_lost_km = km_data.total_lost_km or 0
                        total_planned_km = km_data.total_planned_km or 0
                        if total_planned_km == 0:
                            lost_km_rate = 0
                        else:
                            lost_km_rate = (total_lost_km / total_planned_km) * 100
                        
                        # Query BigQuery for farebox recovery data
                        # Removed cost_per_km from the query since we're using static value
                        fare_query = f"""
                        SELECT 
                            SUM(fare_by_card_passengers) as total_card_fare,
                            SUM(fare_by_cash_passengers) as total_cash_fare,
                            SUM(operated_km) as total_operated_km
                        FROM 
                            {BIGQUERY_DATASET_ID}.tansittable
                        WHERE 
                            FORMAT_DATE('%Y-%m', service_day) = '{month}'
                        """
                        
                        query_job = bq_client.query(fare_query, job_config=job_config)
                        fare_result = query_job.result()
                        fare_data = list(fare_result)[0]
                        
                        # Calculate farebox recovery ratio
                        total_card_fare = fare_data.total_card_fare or 0
                        total_cash_fare = fare_data.total_cash_fare or 0
                        total_operated_km = fare_data.total_operated_km or 0
                        # Using static value for cost per km
                        avg_cost_per_km = 2.0
                        
                        total_fare_revenue = total_card_fare + total_cash_fare
                        total_operating_cost = total_operated_km * avg_cost_per_km
                        
                        if total_operating_cost == 0:
                            farebox_recovery_ratio = 0
                        else:
                            farebox_recovery_ratio = (total_fare_revenue / total_operating_cost) * 100
                        
                        # Create fun performance evaluations with emojis and playful language
                        if seat_occupancy >= 80:
                            occupancy_evaluation = "absolutely stellar! 🌟 Our vehicles are packed like a concert for a chart-topping band!"
                        elif seat_occupancy >= 60:
                            occupancy_evaluation = "rocking it! 🚀 Our seats are getting plenty of love this month."
                        elif seat_occupancy >= 40:
                            occupancy_evaluation = "decent, but not throwing any parties yet. 🎯 We've got room for improvement!"
                        else:
                            occupancy_evaluation = "looking a bit lonely. 😢 Time to jazz up those empty seats!"
                            
                        if lost_km_rate <= 5:
                            km_evaluation = "phenomenal! 🏆 Our vehicles are running so smoothly they might as well be on rails!"
                        elif lost_km_rate <= 10:
                            km_evaluation = "pretty solid! 👍 Just a few detours on our journey to perfection."
                        elif lost_km_rate <= 15:
                            km_evaluation = "so-so. 🤔 We've had better months, but we've had worse too!"
                        else:
                            km_evaluation = "a bit concerning. 🚨 Looks like our vehicles are taking unexpected vacations!"
                        
                        # Add evaluation for farebox recovery ratio
                        if farebox_recovery_ratio >= 70:
                            farebox_evaluation = "outstanding! 💰 We're practically printing money while we drive!"
                        elif farebox_recovery_ratio >= 50:
                            farebox_evaluation = "healthy! 💵 The fare box is keeping up with costs quite nicely."
                        elif farebox_recovery_ratio >= 30:
                            farebox_evaluation = "acceptable. 💸 We're covering some costs, but could use a boost."
                        else:
                            farebox_evaluation = "needs improvement. 📉 Our fare revenue isn't keeping pace with operating costs."
                        
                        # Format month for display
                        month_date = datetime.strptime(month, "%Y-%m")
                        month_name = month_date.strftime("%B %Y")
                        
                        # Get additional metrics for context
                        context_query = f"""
                        SELECT
                            COUNT(DISTINCT route) as total_routes,
                            COUNT(DISTINCT trip_id) as total_trips,
                            AVG(operated_km) as avg_operated_km,
                            SUM(operated_km) as total_operated_km
                        FROM
                            {BIGQUERY_DATASET_ID}.tansittable
                        WHERE
                            FORMAT_DATE('%Y-%m', service_day) = '{month}'
                        """
                        
                        query_job = bq_client.query(context_query, job_config=job_config)
                        context_result = query_job.result()
                        context_data = list(context_result)[0]
                        
                        total_routes = context_data.total_routes or 0
                        total_trips = context_data.total_trips or 0
                        avg_operated_km = context_data.avg_operated_km or 0
                        total_operated_km = context_data.total_operated_km or 0
                        
                        # Add fun facts about the distance
                        distance_fact = ""
                        if total_operated_km > 0:
                            if total_operated_km > 40075:  # Earth's circumference in km
                                earth_trips = total_operated_km / 40075
                                distance_fact = f"Fun fact: Our vehicles traveled enough to circle the Earth {earth_trips:.1f} times! 🌍"
                            elif total_operated_km > 384400:  # Distance to the moon
                                moon_percentage = (total_operated_km / 384400) * 100
                                distance_fact = f"Fun fact: Our vehicles traveled {moon_percentage:.1f}% of the way to the moon! 🌙"
                            else:
                                distance_fact = f"Fun fact: If laid end to end, our routes would stretch across {total_operated_km/1000:.1f} full marathons! 🏃‍♂️"
                        
                        # Generate fun summary with personality
                        summary = {
                            "month": month_name,
                            "title": f"🎉 Transit Spectacular: The {month_name} Edition! 🎉",
                            "intro": f"Hold onto your seats (or maybe not, since our occupancy rate is {seat_occupancy:.2f}%)! Here's your monthly transit breakdown with all the thrills and spills of {month_name}!",
                            "seat_occupancy_rate": f"{seat_occupancy:.2f}%",
                            "seat_occupancy_evaluation": occupancy_evaluation,
                            "lost_km_rate": f"{lost_km_rate:.2f}%",
                            "lost_km_evaluation": km_evaluation,
                            "farebox_recovery_ratio": f"{farebox_recovery_ratio:.2f}%",
                            "farebox_evaluation": farebox_evaluation,
                            "avg_passengers": float(avg_passengers),
                            "avg_seats": float(avg_seats),
                            "total_lost_km": float(total_lost_km),
                            "total_planned_km": float(total_planned_km),
                            "total_operated_km": float(total_operated_km),
                            "total_routes": int(total_routes),
                            "total_trips": int(total_trips),
                            "avg_trip_km": float(avg_operated_km),
                            "total_fare_revenue": float(total_fare_revenue),
                            "total_operating_cost": float(total_operating_cost),
                            "distance_fact": distance_fact,
                            "conclusion": f"That wraps up our {month_name} adventure! Keep those wheels turning! 🚌💨"
                        }
                        
                        api_response = json.dumps(summary)
                        api_requests_and_responses.append(
                            [response.function_call.name, params, api_response]
                        )
                                            
                    print(api_response)                    
                    response = chat.send_message(
                        Part.from_function_response(
                            name=response.function_call.name,
                            response={
                                "content": api_response,
                            },),   )
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
            
            # Check if this is a monthly summary and format it nicely if it is
            if "monthly summary" in prompt.lower() or "this month" in prompt.lower():
                try:
                    # Look for summary data in API responses
                    for log in api_requests_and_responses:
                        if log[0] == "generate_monthly_summary":
                            summary_data = json.loads(log[2])
                            
                            # Create a nicely formatted response with the fun data
                            formatted_response = f"""
                            # {summary_data['title']}
                            
                            {summary_data['intro']}
                            
                            ## The Numbers That Matter
                            
                            🪑 **Seat Occupancy Rate**: {summary_data['seat_occupancy_rate']} - {summary_data['seat_occupancy_evaluation']}
                            
                            🛣️ **Lost Kilometer Rate**: {summary_data['lost_km_rate']} - {summary_data['lost_km_evaluation']}
                            
                            💰 **Farebox Recovery Ratio**: {summary_data['farebox_recovery_ratio']} - {summary_data['farebox_evaluation']}
                            
                            🚌 **Total Trips**: {summary_data['total_trips']} trips across {summary_data['total_routes']} unique routes
                            
                            {summary_data['distance_fact']}
                            
                            ## The Big Picture
                            
                            Average passengers per vehicle: {summary_data['avg_passengers']:.1f}
                            
                            Total distance covered: {summary_data['total_operated_km']:.1f} km
                            
                            Total fare revenue: ${summary_data['total_fare_revenue']:.2f}
                            
                            {summary_data['conclusion']}
                            """
                            
                            # Use our formatted response instead of the model's text response
                            full_response = formatted_response
                            break
                except Exception as e:
                    # If formatting fails, fall back to the model's response
                    print(f"Error formatting fun response: {str(e)}")
                    pass
            
            with message_placeholder.container():
                st.markdown(full_response.replace("$", r"\$"))  # noqa: W605
                
                with st.expander("Function calls, parameters, and responses:"):
                    st.markdown(backend_details)

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": full_response,
                    "backend_details": backend_details,
                   
                }   )
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
                }  )
