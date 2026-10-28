import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.rag_pipeline import RAGPipeline

result = RAGPipeline().query('What is the role of DTI tractography in glioma surgical planning?', 'DEMO_001')
print(result['answer'])
print('Evidence:', len(result['retrieved_evidence']))
print('Warnings:', *result['warnings'], sep='\n- ')
