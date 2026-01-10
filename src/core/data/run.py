from pdf_processor import PDFToPineconePipeline

pipeline = PDFToPineconePipeline(
    index_name="neuroai-index", 
    pinecone_api_key="",
    openai_api_key=""
)

# Definir metadados
metadata = {
    "application": "tea_app",
    "user": "renansantosmendes",
    "patient": "antonio_damasceno",
    "storage_date": "2025-01-10"
}

# Ou diretório
result = pipeline.process_directory("./documents/", metadata=metadata)