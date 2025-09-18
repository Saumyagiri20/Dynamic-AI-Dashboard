from graph import app

if __name__ == "__main__":
    initial_state = {
        "query": "Show revenue trends by product and region for the last year",
        "schema_context": {"tables": ["sales"], "fields": ["revenue", "product", "region", "date"]},
        "session_context": {"previous_queries": [], "user_preferences": {}},
    }
    result = app.invoke(initial_state)
    print("Agent Output:\n", result)
