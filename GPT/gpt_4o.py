import requests
from config.config import API_KEY

headers = {
    "Content-Type": "application/json",
    "api-key": API_KEY,
}


def llm_response(prompt):
    # Payload for the request
    payload = {
        "messages": [
            {
                "role": "system",
                "content": prompt
            }
        ],
        "temperature": 1,
        "top_p": 0.95,
        "max_tokens": 400
    }

    ENDPOINT = "https://genai-jp.openai.azure.com/openai/deployments/ln-gpt40/chat/completions?api-version=2024-02-15-preview"

    # Send request
    try:
        response = requests.post(ENDPOINT, headers=headers, json=payload)
        response.raise_for_status()  # Raise an error for bad HTTP responses
    except requests.RequestException as e:
        raise SystemExit(f"Failed to make the request. Error: {e}")

    # Parse and return the response content
    try:
        return response.json()['choices'][0]['message']['content']
    except KeyError:
        raise ValueError("Unexpected response format: Could not find the 'choices' field in the response.")


# Example usage
if __name__ == "__main__":
    prompt = "Explain the concept of machine learning in simple terms."
    try:
        response = llm_response(prompt)
        print("Response from LLM:")
        print(response)
    except Exception as e:
        print(f"Error: {e}")
