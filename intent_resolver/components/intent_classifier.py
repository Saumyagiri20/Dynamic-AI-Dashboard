import json


class IntentClassifier:
    def __init__(self, llm_handler):
        self.llm_handler = llm_handler

    def classify(self, query, schema_context, session_context, prompt_template):
        prompt = prompt_template.format(
            query=query,
            schema_context=json.dumps(schema_context),
            session_context=json.dumps(session_context)
        )
        llm_output = self.llm_handler.infer(prompt)
        json_start = llm_output.find("{")
        json_end = llm_output.rfind("}") + 1
        try:
            intent_json = json.loads(llm_output[json_start:json_end])
            return {"intent_type": intent_json.get("intent_type", "summary")}
        except Exception:
            return {"intent_type": "summary"}
