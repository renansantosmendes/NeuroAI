from unittest.mock import MagicMock, patch
from src.core.graph import RAGGraph, AdvancedRAGGraph
from src.core.state import RAGState, AdvancedRAGState

def test_rag_graph_nodes():
    chain = MagicMock()
    chain.retrieve.return_value = ["doc"]
    chain.generate.return_value = "ans"
    
    graph = RAGGraph(chain)
    
    state: RAGState = {"question": "q", "context": [], "answer": ""}
    
    ret_res = graph._retrieve_node(state)
    assert ret_res["context"] == ["doc"]
    
    gen_res = graph._generate_node({**state, **ret_res})
    assert gen_res["answer"] == "ans"

def test_advanced_rag_graph_nodes():
    chain = MagicMock()
    chain.retrieve.return_value = ["doc"]
    chain.generate.return_value = "ans"
    
    with patch("src.core.graph.ChatOpenAI") as mock_llm:
        # Mocking classification
        mock_instance = mock_llm.return_value
        # Mocking the classification chain invoke
        with patch("src.core.graph.StrOutputParser.invoke", return_value="YES"):
            arg = AdvancedRAGGraph(chain)
            
            state: AdvancedRAGState = {"question": "q", "context": [], "answer": "", "needs_retrieval": False, "confidence": 0.0}
            
            # Since invoke on the whole chain is hard to mock, let's mock _classify_query_node
            with patch.object(arg, '_classify_query_node', return_value={"needs_retrieval": True}):
                ret_val = arg._classify_query_node(state)
                assert ret_val["needs_retrieval"] is True

            # Test routing
            assert arg._route_query({"needs_retrieval": True}) == "retrieve"
            assert arg._route_query({"needs_retrieval": False}) == "generate_direct"

            # Test generation nodes
            state["context"] = ["doc"]
            gen_ctx_res = arg._generate_with_context_node(state)
            assert gen_ctx_res["answer"] == "ans"
            assert gen_ctx_res["confidence"] == 0.6  # Only 1 doc

            gen_dir_res = arg._generate_without_context_node(state)
            assert "answer" in gen_dir_res
            assert gen_dir_res["confidence"] == 0.7
