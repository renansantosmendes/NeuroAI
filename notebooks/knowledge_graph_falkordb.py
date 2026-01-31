import os
import json
from typing import List, Optional
from dotenv import load_dotenv

# PDF Processing
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# LLM e Chains
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

# Graph
import networkx as nx
from falkordb import FalkorDB

# Load environment variables
load_dotenv()

# --- Data Models ---

class Entity(BaseModel):
    """Representa uma entidade no grafo de conhecimento."""
    name: str = Field(description="Nome da entidade")
    type: str = Field(description="Tipo da entidade (pessoa, organização, conceito, local, evento, tecnologia, etc.)")
    description: Optional[str] = Field(default=None, description="Descrição breve da entidade")


class Relationship(BaseModel):
    """Representa uma relação entre duas entidades."""
    source: str = Field(description="Entidade de origem")
    target: str = Field(description="Entidade de destino")
    relation: str = Field(description="Tipo de relação (ex: trabalha_em, é_parte_de, criou, fundou, etc.)")
    description: Optional[str] = Field(default=None, description="Descrição da relação")


class KnowledgeGraphChunk(BaseModel):
    """Representa as entidades e relações extraídas de um chunk de texto."""
    entities: List[Entity] = Field(description="Lista de entidades identificadas")
    relationships: List[Relationship] = Field(description="Lista de relações entre entidades")


# --- Extractor Class ---

class KnowledgeGraphExtractor:
    def __init__(self, model: str = "gpt-4o-mini", temperature: float = 0):
        self.llm = ChatOpenAI(model=model, temperature=temperature)
        self.parser = PydanticOutputParser(pydantic_object=KnowledgeGraphChunk)
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """Você é um especialista em extração de conhecimento estruturado.
Sua tarefa é extrair entidades e relações de textos para construir um grafo de conhecimento.

REGRAS IMPORTANTES:
1. Identifique TODAS as entidades importantes:
   - Pessoas (nome, cargo, papel)
   - Organizações (empresas, instituições, grupos)
   - Conceitos (ideias, teorias, métodos)
   - Locais (países, cidades, lugares)
   - Eventos (conferências, acontecimentos)
   - Tecnologias (ferramentas, linguagens, frameworks)
   - Produtos (softwares, serviços)
   - Datas (anos, períodos)

2. Para cada relação, use verbos claros como:
   - criou, fundou, desenvolveu
   - trabalha_em, pertence_a
   - é_parte_de, é_tipo_de
   - ocorreu_em, aconteceu_em
   - usa, utiliza, baseia_em

3. Mantenha nomes CONSISTENTES (mesma entidade = mesmo nome)
4. Extraia APENAS informações explícitas no texto
5. Seja ABRANGENTE - capture todas as conexões possíveis

{format_instructions}"""),
            ("human", "Extraia as entidades e relações do seguinte texto:\n\n{text}")
        ])
        self.chain = self.prompt | self.llm | self.parser
        self.graph = nx.DiGraph()
        self.entities = {}  # name (lowercase) -> Entity
        self.relationships = [] # List of Relationship objects

    def load_pdf(self, pdf_path: str, chunk_size: int = 2000, chunk_overlap: int = 200) -> List[str]:
        print(f"📄 Carregando PDF: {pdf_path}")
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        chunks = text_splitter.split_documents(documents)
        return [chunk.page_content for chunk in chunks]

    def extract_from_chunk(self, text: str) -> KnowledgeGraphChunk:
        try:
            result = self.chain.invoke({
                "text": text,
                "format_instructions": self.parser.get_format_instructions()
            })
            return result
        except Exception as e:
            print(f"⚠️ Erro ao processar chunk: {e}")
            return KnowledgeGraphChunk(entities=[], relationships=[])

    def _consolidate(self, chunk: KnowledgeGraphChunk):
        for entity in chunk.entities:
            name_key = entity.name.strip().lower()
            if name_key not in self.entities:
                self.entities[name_key] = entity
                self.graph.add_node(
                    name_key,
                    type=entity.type.lower(),
                    description=entity.description or "",
                    label=entity.name
                )

        for rel in chunk.relationships:
            source_key = rel.source.strip().lower()
            target_key = rel.target.strip().lower()

            if source_key not in self.graph:
                self.graph.add_node(source_key, type="unknown", description="", label=rel.source)
            if target_key not in self.graph:
                self.graph.add_node(target_key, type="unknown", description="", label=rel.target)

            if not self.graph.has_edge(source_key, target_key):
                self.graph.add_edge(
                    source_key,
                    target_key,
                    relation=rel.relation,
                    description=rel.description or ""
                )
                self.relationships.append(rel)

    def extract_from_pdf(self, pdf_path: str, max_chunks: int = None):
        chunks = self.load_pdf(pdf_path)
        if max_chunks:
            chunks = chunks[:max_chunks]
        
        print(f"🔍 Processando {len(chunks)} chunks...")
        for i, chunk in enumerate(chunks, 1):
            print(f"  [{i}/{len(chunks)}] Extraindo...", end=" ")
            kg_chunk = self.extract_from_chunk(chunk)
            self._consolidate(kg_chunk)
            print(f"✓ (+{len(kg_chunk.entities)} entidades, +{len(kg_chunk.relationships)} relações)")

# --- FalkorDB Integration ---

class FalkorDBManager:
    def __init__(self, host=None, port=None, username=None, password=None):
        host = host or os.getenv("FALKORDB_HOST", "localhost")
        port = int(port or os.getenv("FALKORDB_PORT", 6379))
        username = username or os.getenv("FALKORDB_USERNAME")
        password = password or os.getenv("FALKORDB_PASSWORD")
        
        self.db = FalkorDB(host=host, port=port, username=username, password=password)
        print(f"🔌 Conectado ao FalkorDB em {host}:{port}")

    def upsert_knowledge_graph(self, extractor: KnowledgeGraphExtractor, graph_name: str):
        graph = self.db.select_graph(graph_name)
        
        print(f"🚀 Iniciando upsert no grafo '{graph_name}'...")
        
        # 1. Upsert Nodes
        node_count = 0
        for name_key, entity in extractor.entities.items():
            # Escape single quotes for Cypher
            safe_name = entity.name.replace("'", "\\'")
            # Labels can't have spaces or special chars usually, let's normalize
            safe_label = entity.type.replace(" ", "_").replace("-", "_").title()
            if not safe_label: safe_label = "Entity"
            
            description = (entity.description or "").replace("'", "\\'")
            
            # Use name_key as a unique ID to avoid duplicates
            query = f"MERGE (n:{safe_label} {{id: '{name_key}'}}) SET n.name = '{safe_name}', n.description = '{description}'"
            graph.query(query)
            node_count += 1
            
        # 2. Upsert Relationships
        rel_count = 0
        for rel in extractor.relationships:
            source_key = rel.source.strip().lower()
            target_key = rel.target.strip().lower()
            
            # Normalize relationship type
            safe_rel_type = rel.relation.upper().replace(" ", "_").replace("-", "_")
            if not safe_rel_type: safe_rel_type = "RELATED_TO"
            
            description = (rel.description or "").replace("'", "\\'")
            
            # Match by id (which is our name_key)
            query = f"""
            MATCH (a {{id: '{source_key}'}}), (b {{id: '{target_key}'}})
            MERGE (a)-[r:{safe_rel_type}]->(b)
            SET r.description = '{description}'
            """
            graph.query(query)
            rel_count += 1
            
        print(f"✅ Upsert concluído: {node_count} nós e {rel_count} relações processadas.")

# --- Main Execution Example ---

if __name__ == "__main__":
    # 1. Initialize Extractor
    extractor = KnowledgeGraphExtractor(model="gpt-4o-mini")
    
    # 2. Process a PDF or Text
    # SAMPLE_TEXT = "A Inteligência Artificial foi fundada em 1956 por John McCarthy."
    # extractor._consolidate(extractor.extract_from_text(SAMPLE_TEXT))
    
    # Para testar com um PDF real (descomente as linhas abaixo e aponte para um arquivo):
    # pdf_path = "documento.pdf"
    # if os.path.exists(pdf_path):
    #     extractor.extract_from_pdf(pdf_path, max_chunks=5)
    
    # 3. Connect to FalkorDB and Upsert
    try:
        falkor = FalkorDBManager(
            host=None, 
            port=None, 
            username=None, 
            password=None
            )
        falkor.upsert_knowledge_graph(extractor, "neuroai_kg")
    except Exception as e:
        print(f"❌ Erro na integração com FalkorDB: {e}")
        print("Certifique-se de que o FalkorDB está rodando e as credenciais estão no .env ou passadas no construtor.")
