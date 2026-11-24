"""A session-scoped Gradio demo: uploaded documents are not shared across sessions."""

import os

from .classification import train_classifier
from .examples import TRAIN_LABELS, TRAIN_TEXTS


async def _run_agent(text, question):
    import gradio as gr

    if not os.getenv("OPENAI_API_KEY"):
        raise gr.Error("Set OPENAI_API_KEY to use the LLM agent")
    if not text.strip() or not question.strip():
        raise gr.Error("Analyze a document and enter a question first")
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import HumanMessage
    from .retrieval import build_retriever
    from .mcp_tools import make_mcp
    from .agents import build_agent

    graph = build_agent(ChatOpenAI(model="gpt-4o-mini", temperature=0),
                        build_retriever({"uploaded": text}), make_mcp({"uploaded": text}))
    result = await graph.ainvoke({"messages": [HumanMessage(content=question)]},
                                config={"recursion_limit": 8})
    return result["messages"][-1].content


def create_demo():
    import gradio as gr
    from .ocr import extract_text
    from .workflow import build_workflow

    classifier = train_classifier(TRAIN_TEXTS, TRAIN_LABELS)
    workflow = build_workflow(classifier)

    def analyze(image_path, language):
        if not image_path:
            raise gr.Error("Upload an image first")
        text = extract_text(image_path, language)
        category = str(classifier.predict([text])[0]) if text.strip() else "unknown"
        try:
            from .entities import extract_entities
            entities = extract_entities(text, "de_core_news_sm" if language == "de" else "en_core_web_sm")
        except OSError:
            entities = [{"warning": "Install the matching spaCy language model to enable NER"}]
        return text, category, entities, text

    def classify_text(text):
        return workflow.invoke({"text": text}) if text.strip() else {}

    def query(text, question):
        if not text.strip() or not question.strip():
            raise gr.Error("Analyze a document and enter a question first")
        from .retrieval import build_retriever
        from .rag import ask
        return ask(question, build_retriever({"uploaded": text}))

    with gr.Blocks(title="Document AI Toolkit") as demo:
        gr.Markdown("# Document AI Toolkit\nModular OCR, classification, NER, RAG and read-only agents. Use synthetic or approved documents only.")
        doc_state = gr.State("")
        with gr.Tab("OCR / Classification / NER"):
            image = gr.Image(type="filepath", label="Document image")
            lang = gr.Dropdown(["en", "de"], value="en", label="OCR language")
            extracted = gr.Textbox(label="OCR text", lines=9)
            category = gr.Textbox(label="Predicted class")
            entities = gr.JSON(label="Entities")
            gr.Button("Analyze").click(analyze, [image, lang], [extracted, category, entities, doc_state])
        with gr.Tab("LangGraph workflow"):
            workflow_input = gr.Textbox(label="Text", lines=5)
            workflow_result = gr.JSON(label="Workflow state")
            gr.Button("Run").click(classify_text, workflow_input, workflow_result)
        with gr.Tab("LangChain RAG"):
            question = gr.Textbox(label="Question about the uploaded document")
            answer = gr.JSON(label="Answer and source IDs")
            gr.Button("Ask").click(query, [doc_state, question], answer)
        with gr.Tab("LangGraph + MCP agent"):
            agent_question = gr.Textbox(label="Task or question")
            agent_answer = gr.Textbox(label="Agent response")
            gr.Button("Run agent").click(_run_agent, [doc_state, agent_question], agent_answer)
    return demo


def main():
    create_demo().launch()


if __name__ == "__main__":
    main()
