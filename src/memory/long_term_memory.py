"""
Long-Term Memory Module

This module provides the LongTermMemory class which stores persistent knowledge
and lessons learned during research.
"""

from typing import Dict, List, Any, Optional, Union
import json
import os
import time
import uuid
import numpy as np
import faiss

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage, HumanMessage

from .base_memory import BaseMemory


class LongTermMemory(BaseMemory):
    """
    Long-Term Memory stores persistent knowledge and lessons learned.
    
    This memory component provides a vector database implementation for semantic
    storage and retrieval of information with hierarchical organization.
    """
    
    def __init__(
        self,
        llm: BaseChatModel,
        name: str = "LTM",
        dimension: int = 1536,  # Default dimension for embeddings
        storage_path: Optional[str] = None,
        confidence_threshold: float = 0.7,
    ):
        """
        Initialize a LongTermMemory instance.
        
        Args:
            llm: The language model to use for embeddings and operations
            name: The name/identifier for this memory system (default: "LTM")
            dimension: Dimension of embedding vectors (default: 1536)
            storage_path: Path to store the vector database (default: in-memory)
            confidence_threshold: Minimum similarity score threshold (default: 0.7)
        """
        super().__init__(name=name)
        self.llm = llm
        self.dimension = dimension
        self.storage_path = storage_path
        self.confidence_threshold = confidence_threshold
        
        # Initialize index
        self.index = faiss.IndexFlatL2(dimension)
        
        # Dictionary to map IDs to content
        self.id_to_content = {}
        
        # Dictionary for hierarchical organization
        self.hierarchies = {}
        
        # Load existing data if storage path is provided
        if storage_path and os.path.exists(storage_path):
            self._load_from_disk()
    
    async def add(self, key: str, value: Any, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Add an item to long-term memory.
        
        Args:
            key: The key to store the value under
            value: The value to store
            metadata: Optional metadata about the value
            
        Returns:
            Success status
        """
        # Create metadata if none provided
        metadata = metadata or {}
        
        # Add timestamp and confidence if not provided
        if "timestamp" not in metadata:
            metadata["timestamp"] = time.time()
        if "confidence" not in metadata:
            metadata["confidence"] = 1.0  # Default high confidence
        if "id" not in metadata:
            metadata["id"] = str(uuid.uuid4())
        
        # Get parent path for hierarchical organization
        parent_path = metadata.get("parent_path", "")
        
        # Convert value to string if necessary for embedding
        content_str = self._value_to_string(value)
        
        # Create vector embedding
        embedding = await self._create_embedding(content_str)
        
        # Create memory item
        memory_item = {
            "key": key,
            "value": value,
            "metadata": metadata,
            "content_str": content_str,
            "path": f"{parent_path}/{key}" if parent_path else key
        }
        
        # Add to index
        item_id = len(self.id_to_content)
        self.index.add(np.array([embedding], dtype=np.float32))
        self.id_to_content[item_id] = memory_item
        
        # Update hierarchies
        self._update_hierarchy(memory_item)
        
        # Save to disk if storage path is provided
        if self.storage_path:
            self._save_to_disk()
        
        return True
    
    async def get(self, key: str) -> Optional[Any]:
        """
        Retrieve an item from long-term memory by key.
        
        Args:
            key: The key to retrieve
            
        Returns:
            The stored value, or None if not found
        """
        # Search for exact key match
        for item_id, item in self.id_to_content.items():
            if item["key"] == key:
                return item["value"]
        
        # Also check paths for hierarchical retrieval
        for item_id, item in self.id_to_content.items():
            if item["path"] == key:
                return item["value"]
        
        return None
    
    async def search(self, query: Any, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Search long-term memory for relevant items.
        
        Args:
            query: The search query (string, dict, or embedding vector)
            limit: Maximum number of results to return
            
        Returns:
            List of matching items with metadata
        """
        results = []
        
        # If query is already an embedding vector
        if isinstance(query, np.ndarray):
            embedding = query
        
        # If query is a string
        elif isinstance(query, str):
            # First try exact key match
            for item_id, item in self.id_to_content.items():
                if item["key"] == query or item["path"] == query:
                    results.append({
                        "key": item["key"],
                        "value": item["value"],
                        "metadata": item["metadata"],
                        "similarity": 1.0  # Exact match
                    })
            
            # If we have enough exact matches, return them
            if len(results) >= limit:
                return results[:limit]
            
            # Otherwise, create embedding for semantic search
            embedding = await self._create_embedding(query)
        
        # If query is a dict
        elif isinstance(query, dict):
            # Try to match fields
            for item_id, item in self.id_to_content.items():
                match = True
                
                # Check each field in the query
                for field_key, field_value in query.items():
                    # Check if field exists in value (if value is a dict)
                    if isinstance(item["value"], dict) and field_key in item["value"]:
                        if item["value"][field_key] != field_value:
                            match = False
                            break
                    # Check if field exists in metadata
                    elif field_key in item["metadata"]:
                        if item["metadata"][field_key] != field_value:
                            match = False
                            break
                    # Check if field is path or key
                    elif field_key == "path" and item["path"] == field_value:
                        continue
                    elif field_key == "key" and item["key"] == field_value:
                        continue
                    # No match for this field
                    else:
                        match = False
                        break
                
                if match:
                    results.append({
                        "key": item["key"],
                        "value": item["value"],
                        "metadata": item["metadata"],
                        "similarity": 1.0  # Exact match
                    })
            
            # If we have enough matches, return them
            if len(results) >= limit:
                return results[:limit]
            
            # Otherwise, create embedding for semantic search using string representation
            query_str = json.dumps(query)
            embedding = await self._create_embedding(query_str)
        
        else:
            # Unsupported query type
            return []
        
        # Perform semantic search
        D, I = self.index.search(np.array([embedding], dtype=np.float32), limit)
        
        # Process search results
        for i, (distance, item_id) in enumerate(zip(D[0], I[0])):
            if item_id < 0 or item_id >= len(self.id_to_content):
                continue
            
            # Calculate similarity score (convert distance to similarity)
            similarity = 1.0 / (1.0 + distance)
            
            # Skip if below confidence threshold
            if similarity < self.confidence_threshold:
                continue
            
            # Check if already in results from exact match
            item = self.id_to_content[item_id]
            if any(r["key"] == item["key"] for r in results):
                continue
            
            results.append({
                "key": item["key"],
                "value": item["value"],
                "metadata": item["metadata"],
                "similarity": similarity
            })
        
        # Sort by similarity (highest first)
        results.sort(key=lambda x: -x["similarity"])
        
        return results[:limit]
    
    async def update(self, key: str, value: Any, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Update an existing item in long-term memory.
        
        Args:
            key: The key of the item to update
            value: The new value
            metadata: Optional new metadata
            
        Returns:
            Success status
        """
        # Find item by key
        found = False
        item_id_to_update = None
        old_path = None
        
        for item_id, item in self.id_to_content.items():
            if item["key"] == key:
                item_id_to_update = item_id
                old_path = item["path"]
                found = True
                break
        
        if not found:
            return False
        
        # Get current item
        current_item = self.id_to_content[item_id_to_update]
        
        # Get parent path for hierarchical organization
        parent_path = metadata.get("parent_path", "") if metadata else current_item["metadata"].get("parent_path", "")
        
        # Update value
        current_item["value"] = value
        
        # Update metadata if provided
        if metadata:
            current_item["metadata"].update(metadata)
        
        # Update timestamp
        current_item["metadata"]["timestamp"] = time.time()
        
        # Update content string
        content_str = self._value_to_string(value)
        current_item["content_str"] = content_str
        
        # Update path
        new_path = f"{parent_path}/{key}" if parent_path else key
        current_item["path"] = new_path
        
        # Update embedding
        embedding = await self._create_embedding(content_str)
        
        # Update index (requires removing and re-adding)
        # This is a simplified approach - in production, might use a more efficient method
        vectors = []
        for i in range(len(self.id_to_content)):
            if i == item_id_to_update:
                vectors.append(embedding)
            else:
                # Create dummy search to get the vector
                D, I = self.index.search(np.array([np.zeros(self.dimension)], dtype=np.float32), 1)
                vectors.append(np.zeros(self.dimension))  # Placeholder
        
        # Rebuild index
        self.index = faiss.IndexFlatL2(self.dimension)
        self.index.add(np.array(vectors, dtype=np.float32))
        
        # Update hierarchies
        if old_path != new_path:
            self._rebuild_hierarchies()
        else:
            self._update_hierarchy(current_item)
        
        # Save to disk if storage path is provided
        if self.storage_path:
            self._save_to_disk()
        
        return True
    
    async def delete(self, key: str) -> bool:
        """
        Delete an item from long-term memory.
        
        Args:
            key: The key of the item to delete
            
        Returns:
            Success status
        """
        # Find item by key
        found = False
        item_id_to_delete = None
        
        for item_id, item in self.id_to_content.items():
            if item["key"] == key:
                item_id_to_delete = item_id
                found = True
                break
        
        if not found:
            return False
        
        # Remove from hierarchies
        path = self.id_to_content[item_id_to_delete]["path"]
        self._remove_from_hierarchy(path)
        
        # Remove from index and rebuild (simplified approach)
        # In production, might use a more efficient method
        vectors = []
        new_id_to_content = {}
        new_id = 0
        
        for item_id, item in self.id_to_content.items():
            if item_id != item_id_to_delete:
                # Create dummy search to get the vector
                # This is a placeholder - in production, would store vectors separately
                D, I = self.index.search(np.array([np.zeros(self.dimension)], dtype=np.float32), 1)
                vectors.append(np.zeros(self.dimension))  # Placeholder
                
                new_id_to_content[new_id] = item
                new_id += 1
        
        # Rebuild index
        self.index = faiss.IndexFlatL2(self.dimension)
        if vectors:
            self.index.add(np.array(vectors, dtype=np.float32))
        
        # Update content mapping
        self.id_to_content = new_id_to_content
        
        # Save to disk if storage path is provided
        if self.storage_path:
            self._save_to_disk()
        
        return True
    
    async def clear(self) -> bool:
        """
        Clear all items from long-term memory.
        
        Returns:
            Success status
        """
        # Reset index
        self.index = faiss.IndexFlatL2(self.dimension)
        
        # Clear content mapping
        self.id_to_content = {}
        
        # Clear hierarchies
        self.hierarchies = {}
        
        # Save empty state to disk if storage path is provided
        if self.storage_path:
            self._save_to_disk()
        
        return True
    
    async def get_by_path(self, path: str) -> Optional[Any]:
        """
        Retrieve an item from long-term memory by its hierarchical path.
        
        Args:
            path: The path to retrieve
            
        Returns:
            The stored value, or None if not found
        """
        for item_id, item in self.id_to_content.items():
            if item["path"] == path:
                return item["value"]
        
        return None
    
    async def get_children(self, path: str) -> List[Dict[str, Any]]:
        """
        Get all direct children of a path in the hierarchy.
        
        Args:
            path: The parent path
            
        Returns:
            List of child items with their values and metadata
        """
        children = []
        
        for item_id, item in self.id_to_content.items():
            item_path = item["path"]
            parent_path = '/'.join(item_path.split('/')[:-1])
            
            if parent_path == path:
                children.append({
                    "key": item["key"],
                    "value": item["value"],
                    "metadata": item["metadata"],
                    "path": item_path
                })
        
        return children
    
    async def get_hierarchy(self, path: str = "") -> Dict[str, Any]:
        """
        Get a hierarchical view of memory starting from a path.
        
        Args:
            path: The starting path (default: root)
            
        Returns:
            Hierarchical structure of items
        """
        if path in self.hierarchies:
            return self.hierarchies[path]
        
        return {}
    
    async def consolidate(self) -> bool:
        """
        Perform memory consolidation to organize and optimize storage.
        
        This process identifies patterns, removes redundancies, and
        strengthens important connections.
        
        Returns:
            Success status
        """
        # This is a simplified implementation of memory consolidation
        # In a production system, this would involve more sophisticated
        # algorithms for clustering, redundancy detection, etc.
        
        # 1. Identify similar items and cluster them
        all_items = list(self.id_to_content.values())
        
        # Skip if not enough items
        if len(all_items) < 5:
            return False
        
        # 2. Generate summaries for clusters (simplified)
        # Here we're just asking the LLM to summarize a random batch
        # In production, would use actual clustering algorithms
        
        # Select a random batch
        import random
        batch_size = min(5, len(all_items))
        batch = random.sample(all_items, batch_size)
        
        # Extract content for summarization
        batch_content = "\n\n".join([item["content_str"] for item in batch])
        
        # Generate summary
        summary = await self._generate_summary(batch_content)
        
        # 3. Store the summary as a new memory item
        await self.add(
            key=f"summary_{int(time.time())}",
            value=summary,
            metadata={
                "is_summary": True,
                "source_count": batch_size,
                "confidence": 0.8,
                "parent_path": "summaries"
            }
        )
        
        return True
    
    def _value_to_string(self, value: Any) -> str:
        """
        Convert a value to string format for embedding.
        
        Args:
            value: Value to convert
            
        Returns:
            String representation
        """
        if isinstance(value, str):
            return value
        
        try:
            return json.dumps(value)
        except:
            return str(value)
    
    async def _create_embedding(self, text: str) -> np.ndarray:
        """
        Create a vector embedding for text using the LLM's embedding capabilities.
        
        Args:
            text: Text to embed
            
        Returns:
            Vector embedding
        """
        try:
            # Check if we can use the LLM's embedding method directly
            if hasattr(self.llm, 'get_embedding') and callable(getattr(self.llm, 'get_embedding')):
                embedding = await self.llm.aget_embedding(text)
                return np.array(embedding, dtype=np.float32)
            
            # For OpenAI-based LLMs, try to use their embedding models
            elif hasattr(self.llm, 'client') and hasattr(self.llm.client, 'embeddings'):
                from openai.embeddings_utils import get_embedding
                embedding = await self.llm.client.embeddings.create(
                    input=text,
                    model="text-embedding-3-small"
                )
                return np.array(embedding.data[0].embedding, dtype=np.float32)
                
            # For Anthropic Claude models, use a text-to-embedding prompt as workaround
            elif "anthropic" in str(type(self.llm)).lower():
                messages = [
                    SystemMessage(content="You are an embedding generator. Extract the key semantic features from the text and represent them as a comma-separated list of 100 numbers between -1 and 1. Each number should represent a different semantic dimension."),
                    HumanMessage(content=f"Generate a semantic embedding for the following text: {text[:1000]}")
                ]
                
                response = await self.llm.ainvoke(messages)
                # Extract numbers from response
                import re
                numbers = re.findall(r'-?\d+\.\d+', response.content)
                if len(numbers) < self.dimension:
                    # Pad with zeros if too few numbers
                    numbers.extend([0] * (self.dimension - len(numbers)))
                elif len(numbers) > self.dimension:
                    # Truncate if too many numbers
                    numbers = numbers[:self.dimension]
                    
                return np.array([float(n) for n in numbers], dtype=np.float32)
                
            # Fallback to random vector if no embedding method available
            else:
                print("Warning: No embedding method available, using random vectors. This will impact semantic search quality.")
                return np.random.rand(self.dimension).astype(np.float32)
                
        except Exception as e:
            print(f"Error creating embedding: {str(e)}. Using random vector as fallback.")
            return np.random.rand(self.dimension).astype(np.float32)
    
    async def _generate_summary(self, content: str) -> str:
        """
        Generate a summary of content using the LLM.
        
        Args:
            content: Content to summarize
            
        Returns:
            Generated summary
        """
        messages = [
            SystemMessage(content="You are a summarization assistant that creates concise summaries of information for long-term memory storage."),
            HumanMessage(content=f"""
Summarize the following information into a concise, informative summary that captures the key points.
The summary should be useful for future reference and retrieval.

CONTENT TO SUMMARIZE:
{content}

Your summary should:
1. Be concise but comprehensive
2. Capture key facts, concepts, and relationships
3. Organize information logically
4. Be useful for future retrieval and reference
""")
        ]
        
        response = await self.llm.ainvoke(messages)
        return response.content
    
    def _update_hierarchy(self, item: Dict[str, Any]) -> None:
        """
        Update hierarchical organization with an item.
        
        Args:
            item: Memory item to add to hierarchy
        """
        path = item["path"]
        path_parts = path.split("/")
        
        # Initialize current level to root of hierarchy
        current = self.hierarchies
        
        # Navigate and create path as needed
        for i, part in enumerate(path_parts):
            if part not in current:
                current[part] = {}
            
            # If it's the final part, store content reference
            if i == len(path_parts) - 1:
                current[part]["_content"] = {
                    "key": item["key"],
                    "metadata": item["metadata"]
                }
            
            current = current[part]
    
    def _remove_from_hierarchy(self, path: str) -> None:
        """
        Remove an item from the hierarchy by path.
        
        Args:
            path: Path to remove
        """
        path_parts = path.split("/")
        
        # For simplicity, rebuild the hierarchy
        # In production, could use a more efficient approach
        self._rebuild_hierarchies()
    
    def _rebuild_hierarchies(self) -> None:
        """
        Rebuild the complete hierarchical organization.
        """
        self.hierarchies = {}
        
        for item in self.id_to_content.values():
            self._update_hierarchy(item)
    
    def _save_to_disk(self) -> None:
        """
        Save memory state to disk.
        """
        if not self.storage_path:
            return
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
        
        # Prepare data for saving
        save_data = {
            "items": self.id_to_content,
            "hierarchies": self.hierarchies
        }
        
        # Save content data
        with open(self.storage_path, 'w') as f:
            json.dump(save_data, f)
        
        # Save index separately (if not empty)
        if len(self.id_to_content) > 0:
            index_path = f"{self.storage_path}.index"
            faiss.write_index(self.index, index_path)
    
    def _load_from_disk(self) -> None:
        """
        Load memory state from disk.
        """
        if not self.storage_path or not os.path.exists(self.storage_path):
            return
        
        # Load content data
        with open(self.storage_path, 'r') as f:
            try:
                save_data = json.load(f)
                self.id_to_content = save_data.get("items", {})
                self.hierarchies = save_data.get("hierarchies", {})
            except json.JSONDecodeError:
                print(f"Error loading memory state from {self.storage_path}")
                return
        
        # Load index if it exists
        index_path = f"{self.storage_path}.index"
        if os.path.exists(index_path):
            try:
                self.index = faiss.read_index(index_path)
            except:
                print(f"Error loading index from {index_path}")
                # Rebuild index from scratch
                self.index = faiss.IndexFlatL2(self.dimension)