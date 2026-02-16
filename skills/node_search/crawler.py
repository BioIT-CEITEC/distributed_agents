import os
import glob
from typing import List, Dict, Any
from pathlib import Path
import pandas as pd
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import logging

# Configure logging
logger = logging.getLogger(__name__)

try:
    from fastembed import TextEmbedding
    FASTEMBED_AVAILABLE = True
except ImportError:
    FASTEMBED_AVAILABLE = False
    logger.warning("fastembed not available. Semantic search will be disabled.")
except Exception as e:
    FASTEMBED_AVAILABLE = False
    logger.warning(f"Error importing fastembed: {e}. Semantic search will be disabled.")

class Crawler:
    """
    Crawls a directory for files, extracts metadata, 
    and indexes them in an in-memory Qdrant instance for semantic search.
    """
    def __init__(self, root_dir: str):
        self.root_dir = root_dir
        self.client = QdrantClient(":memory:")
        self.collection_name = "node_files"
        self.embedding_enabled = FASTEMBED_AVAILABLE
        
        if self.embedding_enabled:
            try:
                self.embedding_model = TextEmbedding()
                # Initialize Qdrant collection
                self.client.recreate_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(size=384, distance=Distance.COSINE), # 384 for fastembed/BAAI/bge-small-en-v1.5
                )
            except Exception as e:
                logger.error(f"Failed to initialize embedding model: {e}")
                self.embedding_enabled = False
        
        if self.embedding_enabled:
            self._index_files()
        else:
            logger.warning("Skipping file indexing as embedding is disabled.")

    def _index_files(self):
        """Recursively scans and indexes files."""
        if not self.embedding_enabled:
            return

        files = []
        # Support multiple extensions
        extensions = ['*.csv', '*.pdf', '*.docx']
        for ext in extensions:
            files.extend(glob.glob(os.path.join(self.root_dir, '**', ext), recursive=True))
            
        points = []
        for idx, file_path in enumerate(files):
            try:
                metadata = self._extract_metadata(file_path)
                # Create a descriptive text for embedding (Metadata ONLY)
                text_to_embed = f"File: {os.path.basename(file_path)}. Type: {metadata['type']}. Context: {metadata['summary']}"
                
                # Generate embedding
                embedding = list(self.embedding_model.embed([text_to_embed]))[0]
                
                points.append(PointStruct(
                    id=idx,
                    vector=embedding.tolist(),
                    payload={
                        "path": os.path.abspath(file_path),
                        "name": os.path.basename(file_path),
                        "metadata": metadata
                    }
                ))
            except Exception as e:
                logger.error(f"Error indexing file {file_path}: {e}")
            
        if points:
            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )
            print(f"Indexed {len(points)} files in {self.root_dir}")

    def close(self):
        """Closes the Qdrant client."""
        if self.client:
            # Qdrant local/memory client doesn't strictly require close, but good practice
            # and required by the "Ephemeral Vector DB Lifecycle" rule.
            self.client = None
            logger.info("Qdrant client closed.")

    def _extract_metadata(self, file_path: str) -> Dict[str, Any]:
        """Extracts metadata based on file type. SAFE: No data content."""
        ext = os.path.splitext(file_path)[1].lower()
        summary = ""
        
        try:
            if ext == '.csv':
                # Read only ID/header info, NO data rows for embedding
                df = pd.read_csv(file_path, nrows=0) 
                headers = ", ".join(df.columns.tolist())
                # meaningful summary WITHOUT data
                summary = f"CSV file. Columns: {headers}."
            elif ext == '.pdf':
                summary = f"PDF document. Title: {os.path.basename(file_path)}"
            elif ext == '.docx':
                summary = f"Word document. Title: {os.path.basename(file_path)}"
            else:
                summary = "Unknown file type."
        except Exception as e:
            summary = f"Error reading file metadata: {str(e)}"
            
        return {
            "type": ext,
            "summary": summary
        }

    def semantic_search(self, query: str, limit: int = 5) -> List[str]:
        """Performs semantic search on indexed files."""
        if not self.embedding_enabled:
            logger.warning("Semantic search is disabled.")
            return []

        try:
            query_embedding = list(self.embedding_model.embed([query]))[0]
            
            search_result = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding.tolist(),
                limit=limit
            )
            
            return [hit.payload['path'] for hit in search_result]
        except Exception as e:
            logger.error(f"Error during semantic search: {e}")
            return []

if __name__ == "__main__":
    # Test
    import tempfile
    try:
        with tempfile.TemporaryDirectory() as tmpdirname:
            # Create dummy file
            with open(os.path.join(tmpdirname, "test.csv"), "w") as f:
                f.write("col1,col2\n1,2")
                
            crawler = Crawler(tmpdirname)
            if crawler.embedding_enabled:
                results = crawler.semantic_search("data with col1")
                print(results)
            else:
                print("Embedding disabled, skipping test.")
    except Exception as e:
        print(f"Test failed: {e}")
