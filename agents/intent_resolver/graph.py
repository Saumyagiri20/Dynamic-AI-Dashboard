from langgraph.graph import StateGraph, START, END
from components.llm_handler import LLMHandler
from components.intent_classifier import IntentClassifier
from components.metric_extractor import MetricExtractor
from components.dimension_extractor import DimensionExtractor
from components.chart_recommender import ChartRecommender
from components.prompt_enhancer import PromptEnhancer
import json

llm = LLMHandler()

intent_classifier = IntentClassifier(llm)
metric_extractor = MetricExtractor(llm)
dimension_extractor = DimensionExtractor(llm)
chart_recommender = ChartRecommender(llm)
prompt_enhancer = PromptEnhancer(llm)

with open("prompts/intent_classification_prompt.txt") as f:
    prompt_intent = f.read()
with open("prompts/metric_extraction_prompt.txt") as f:
    prompt_metric = f.read()
with open("prompts/dimension_extraction_prompt.txt") as f:
    prompt_dimension = f.read()
with open("prompts/chart_recommendation_prompt.txt") as f:
    prompt_chart = f.read()
with open("prompts/prompt_enhancement_prompt.txt") as f:
    prompt_enhance = f.read()


def classify_intent_node(state):
    result = intent_classifier.classify(
        state.get("query", ""),
        state.get("schema_context", {}),
        state.get("session_context", {}),
        prompt_intent,
    )
    state.update(result)
    return state


def extract_metric_node(state):
    result = metric_extractor.extract_metric(
        state.get("query", ""),
        state.get("schema_context", {}),
        state.get("session_context", {}),
        prompt_metric,
    )
    state.update(result)
    return state


def extract_dimension_node(state):
    result = dimension_extractor.extract_dimensions(
        state.get("query", ""),
        state.get("schema_context", {}),
        state.get("session_context", {}),
        prompt_dimension,
    )
    state.update(result)
    return state


def recommend_chart_node(state):
    result = chart_recommender.recommend_chart(
        state.get("query", ""),
        state.get("schema_context", {}),
        state.get("session_context", {}),
        prompt_chart,
    )
    state.update(result)
    return state


def enhance_prompt_node(state):
    result = prompt_enhancer.enhance(
        state.get("query", ""),
        state.get("schema_context", {}),
        state.get("session_context", {}),
        prompt_enhance,
    )
    state.update(result)
    return state


workflow = StateGraph(dict)
workflow.add_node("classify_intent", classify_intent_node)
workflow.add_node("extract_metric", extract_metric_node)
workflow.add_node("extract_dimension", extract_dimension_node)
workflow.add_node("recommend_chart", recommend_chart_node)
workflow.add_node("enhance_prompt", enhance_prompt_node)

workflow.add_edge(START, "classify_intent")
workflow.add_edge("classify_intent", "extract_metric")
workflow.add_edge("extract_metric", "extract_dimension")
workflow.add_edge("extract_dimension", "recommend_chart")
workflow.add_edge("recommend_chart", "enhance_prompt")
workflow.add_edge("enhance_prompt", END)

app = workflow.compile()
