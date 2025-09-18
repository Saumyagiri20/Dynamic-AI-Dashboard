import json

class PromptEnhancer:
    def __init__(self, llm_handler):
        self.llm_handler = llm_handler

    def enhance(self, query, schema_context, session_context, prompt_template):
        prompt = prompt_template.format(
            query=query,
            schema_context=json.dumps(schema_context),
            session_context=json.dumps(session_context),
        )
        llm_output = self.llm_handler.infer(prompt)
        return {"enhanced_prompt": llm_output.strip()}
