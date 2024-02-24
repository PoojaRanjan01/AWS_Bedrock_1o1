"""
Using llama2
"""
import boto3
import json

# flag_use_llama = False
flag_use_llama = True

def invoke_llama2(prompt):
    """
    Invokes the Meta Llama 2 large-language model to run an inference
    using the input provided in the request body.

    :param prompt: The prompt that you want Llama 2 to complete.
    :return: Inference response from the model.
    """

    try:
        # The different model providers have individual request and response formats.
        # For the format, ranges, and default values for Meta Llama 2 Chat, refer to:
        # https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-meta.html
        body = {
            "prompt": prompt,
            "temperature": 0.5,
            "top_p": 0.9,
            "max_gen_len": 512,
        }

        bedrock_runtime_client = boto3.client(service_name="bedrock-runtime", region_name="us-east-1")

        response = bedrock_runtime_client.invoke_model(
            modelId="meta.llama2-13b-chat-v1", body=json.dumps(body)
        )

        response_body = json.loads(response["body"].read())
        completion = response_body["generation"]

        return completion
    except:
        print("Couldn't invoke Llama 2")
        raise

# Function to generate budget
def s1_generate_budget(origin_country, travel_dates_start, travel_dates_end, destination_country,
                             destination_cities, num_adults, num_kids):
    """
    Query llama model for budget plan
    """
    prompt_1 = f"Generate an estimated budget for a trip to " \
               f"{destination_country} from " \
               f"{origin_country} during {travel_dates_start} " \
               f"to {travel_dates_end}, for {num_adults} adults and" \
               f" {num_kids} kids visiting the {destination_cities}. " \
               f"Give the expenses in the {origin_country} currency and also similar translated in" \
               f" {destination_country} currency."
    response_1 = invoke_llama2(prompt_1) if flag_use_llama else prompt_1
    return response_1


# Function to suggest activities
def s2_suggest_activities(destination_country, travel_dates_start, travel_dates_end, destination_cities,
                                num_adults, num_kids):
    """
    Query llama model for suggested activities
    """
    prompt_2 = f"Suggest top activities and things to do aptly suggested per day in sequence in" \
               f" {destination_country} during {travel_dates_start} to {travel_dates_end}, " \
               f"visiting {destination_cities} for {num_adults} adults and {num_kids} kids. " \
               f"Whenever possible give a link to get more details on the activity."
    response_2 = invoke_llama2(prompt_2) if flag_use_llama else prompt_2
    return response_2
