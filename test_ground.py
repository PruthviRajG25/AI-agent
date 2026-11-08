from openai import OpenAI
import json

client = OpenAI(base_url='http://127.0.0.1:11434/v1', api_key='dummy')

messages = [
    {
        'role': 'system',
        'content': (
            'You are a fast, factual assistant. Always synthesize and base your answer directly '
            'on the live search results returned by the web_search tool. '
            'Never claim your knowledge is cutoff or that an event has not occurred if the search results contain the information.'
        )
    },
    {
        'role': 'user',
        'content': 'Who won the 2024 US presidential election?'
    },
    {
        'role': 'assistant',
        'content': '',
        'tool_calls': [
            {
                'id': 'call_1',
                'type': 'function',
                'function': {
                    'name': 'web_search',
                    'arguments': json.dumps({'query': '2024 US presidential election winner'})
                }
            }
        ]
    },
    {
        'role': 'tool',
        'tool_call_id': 'call_1',
        'content': (
            'Search Results:\n'
            '- 2024 United States presidential election: The Republican ticket of former president Donald Trump and '
            'Ohio senator JD Vance defeated the Democratic ticket of Kamala Harris and Tim Walz.'
        )
    }
]

resp = client.chat.completions.create(model='llama3.2:latest', messages=messages, stream=False)
print('Model answer:\n', resp.choices[0].message.content)
