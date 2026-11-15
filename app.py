"""Gradio Space: serialized persistent pipeline; verbose JSON hidden unless checked."""
import json,os,threading
import gradio as gr
from runtime import Runtime
_runtime=None;lock=threading.RLock()
def predict(question,verbose):
    global _runtime
    with lock:
        if _runtime is None:
            _runtime=Runtime(os.getenv('ARTIFACT_DIR','artifacts'),os.getenv('DEVICE','cuda'),
                float(os.getenv('ORIGINAL_THRESHOLD','.9')),float(os.getenv('FINAL_THRESHOLD','.75')))
        result=_runtime.answer(question)
        compact=_runtime.compact(result)
        original=result.get('original_tag_output','')
        selected=result.get('selected_rag',{}).get('answer') or '(retrieval not selected)'
        final=result.get('final_answer','abstain')
        score=compact.get('final_score')
        score_text='Unavailable' if score is None else f'{score:.4f} (heuristic decision index, not correctness probability)'
        sources='\n'.join(f"{x.get('title','Source')}: {x.get('url','')}" for x in result.get('sources',[])) or 'No retrieval sources used'
        return original,selected,final,score_text,sources,compact,gr.update(value=compact if verbose else None,visible=bool(verbose))
def toggle(verbose,state):return gr.update(value=state if verbose else None,visible=bool(verbose))
with gr.Blocks(title='TAG probe and evidence QA') as demo:
    gr.Markdown('# TAG probe and evidence QA\nResearch demo. Questions may be sent to public Wikipedia. Do not enter confidential data. Scores are not calibrated truth probabilities.')
    q=gr.Textbox(label='Question',lines=2);verbose=gr.Checkbox(label='Verbose: show complete result.json',value=False)
    button=gr.Button('Answer');ma=gr.Textbox(label='Original TAG output');rag=gr.Textbox(label='Selected retrieval answer')
    pa=gr.Textbox(label='Final output');score=gr.Textbox(label='Final score');sources=gr.Textbox(label='Sources')
    state=gr.State({});details=gr.JSON(label='Complete result.json',visible=False)
    button.click(predict,[q,verbose],[ma,rag,pa,score,sources,state,details],concurrency_limit=1)
    verbose.change(toggle,[verbose,state],details)
if __name__=='__main__':demo.queue(max_size=8).launch(server_name='0.0.0.0',server_port=8888,show_error=False)
