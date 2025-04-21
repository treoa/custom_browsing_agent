"""
Working Memory Module

This module provides the WorkingMemory class which actively processes and manipulates
information currently being researched.
"""

from typing import Dict, List, Any, Optional, Set
import time
import json

from .base_memory import BaseMemory


class WorkingMemory(BaseMemory):
    """
    Working Memory actively processes information currently being researched.
    
    This memory component provides temporary storage for active comparison and
    integration of information, with attention mechanisms for focus management.
    """
    
    def __init__(
        self,
        name: str = "WorkingMemory",
        capacity: int = 7,  # Default based on cognitive science "7±2" principle
    ):
        """
        Initialize a WorkingMemory instance.
        
        Args:
            name: The name/identifier for this memory system (default: "WorkingMemory")
            capacity: Maximum number of chunks to store (default: 7)
        """
        super().__init__(name=name)
        self.capacity = capacity
        self._memory = {}
        self._attention_focus = set()  # Set of keys currently in focus
        self._relationships = {}  # Mapping of relationships between items
    
    async def add(self, key: str, value: Any, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Add an item to working memory.
        
        Args:
            key: The key to store the value under
            value: The value to store
            metadata: Optional metadata about the value
            
        Returns:
            Success status
        """
        # Create metadata if none provided
        metadata = metadata or {}
        
        # Add timestamp and attention score if not provided
        if "timestamp" not in metadata:
            metadata["timestamp"] = time.time()
        if "attention_score" not in metadata:
            metadata["attention_score"] = 5  # Default mid-level attention (1-10)
        
        # Create memory item
        memory_item = {
            "key": key,
            "value": value,
            "metadata": metadata,
            "chunk_id": len(self._memory) + 1  # Assign a chunk ID
        }
        
        # Add to memory
        self._memory[key] = memory_item
        
        # Auto-focus on new items with high attention score
        if metadata.get("attention_score", 0) >= 7:
            self._attention_focus.add(key)
        
        # Check if we're over capacity
        if len(self._memory) > self.capacity:
            # Remove items with lowest attention scores
            self._prune_by_attention()
        
        return True
    
    async def get(self, key: str) -> Optional[Any]:
        """
        Retrieve an item from working memory by key.
        
        Args:
            key: The key to retrieve
            
        Returns:
            The stored value, or None if not found
        """
        # Check if key exists
        if key not in self._memory:
            return None
        
        # Get the memory item
        memory_item = self._memory[key]
        
        # Update access timestamp
        memory_item["metadata"]["last_accessed"] = time.time()
        
        # Increase attention score when accessed
        memory_item["metadata"]["attention_score"] = min(
            10, memory_item["metadata"].get("attention_score", 5) + 1
        )
        
        # Add to focus when accessed
        self._attention_focus.add(key)
        
        # If focus is too large, remove items with lowest attention scores
        if len(self._attention_focus) > self.capacity / 2:
            self._prune_focus()
        
        return memory_item["value"]
    
    async def search(self, query: Any, limit: int = 3) -> List[Dict[str, Any]]:
        """
        Search working memory for relevant items.
        
        Args:
            query: The search query (string or dict with field queries)
            limit: Maximum number of results to return
            
        Returns:
            List of matching items with metadata
        """
        results = []
        
        # Simple string search in serialized form
        if isinstance(query, str):
            query_lower = query.lower()
            
            for key, item in self._memory.items():
                # Try to serialize the value for string search
                try:
                    item_str = json.dumps(item["value"]).lower()
                    key_str = str(key).lower()
                    
                    # Check if query matches in key or value
                    if query_lower in key_str or query_lower in item_str:
                        results.append({
                            "key": key,
                            "value": item["value"],
                            "metadata": item["metadata"],
                            "in_focus": key in self._attention_focus
                        })
                except:
                    # If serialization fails, just search the key
                    if query_lower in str(key).lower():
                        results.append({
                            "key": key,
                            "value": item["value"],
                            "metadata": item["metadata"],
                            "in_focus": key in self._attention_focus
                        })
        
        # Dict-based field matching
        elif isinstance(query, dict):
            for key, item in self._memory.items():
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
                    # No match for this field
                    else:
                        match = False
                        break
                
                if match:
                    results.append({
                        "key": key,
                        "value": item["value"],
                        "metadata": item["metadata"],
                        "in_focus": key in self._attention_focus
                    })
        
        # Prioritize items in focus and with higher attention scores
        results.sort(key=lambda x: (
            -int(x["in_focus"]),  # Focused items first
            -x["metadata"].get("attention_score", 0),
            -x["metadata"].get("timestamp", 0)
        ))
        
        # Increase attention for returned items
        for result in results[:limit]:
            key = result["key"]
            if key in self._memory:
                self._memory[key]["metadata"]["attention_score"] = min(
                    10, self._memory[key]["metadata"].get("attention_score", 5) + 0.5
                )
                self._attention_focus.add(key)
        
        # If focus is too large, prune it
        if len(self._attention_focus) > self.capacity / 2:
            self._prune_focus()
        
        # Limit results
        return results[:limit]
    
    async def update(self, key: str, value: Any, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Update an existing item in working memory.
        
        Args:
            key: The key of the item to update
            value: The new value
            metadata: Optional new metadata
            
        Returns:
            Success status
        """
        # Check if key exists
        if key not in self._memory:
            return False
        
        # Get current item
        current_item = self._memory[key]
        
        # Update value
        current_item["value"] = value
        
        # Update metadata if provided
        if metadata:
            current_item["metadata"].update(metadata)
        
        # Update timestamp
        current_item["metadata"]["timestamp"] = time.time()
        
        # Increase attention when updated
        current_item["metadata"]["attention_score"] = min(
            10, current_item["metadata"].get("attention_score", 5) + 1
        )
        
        # Add to focus when updated
        self._attention_focus.add(key)
        
        # If focus is too large, prune it
        if len(self._attention_focus) > self.capacity / 2:
            self._prune_focus()
        
        return True
    
    async def delete(self, key: str) -> bool:
        """
        Delete an item from working memory.
        
        Args:
            key: The key of the item to delete
            
        Returns:
            Success status
        """
        if key in self._memory:
            # Remove from memory
            del self._memory[key]
            
            # Remove from focus
            if key in self._attention_focus:
                self._attention_focus.remove(key)
            
            # Remove relationships
            if key in self._relationships:
                del self._relationships[key]
            
            # Remove from other items' relationships
            for rel_key, related_items in self._relationships.items():
                if key in related_items:
                    related_items.remove(key)
            
            return True
        return False
    
    async def clear(self) -> bool:
        """
        Clear all items from working memory.
        
        Returns:
            Success status
        """
        self._memory.clear()
        self._attention_focus.clear()
        self._relationships.clear()
        return True
    
    async def add_relationship(self, key1: str, key2: str, relationship_type: str = "related") -> bool:
        """
        Add a relationship between two items in working memory.
        
        Args:
            key1: First item key
            key2: Second item key
            relationship_type: Type of relationship (default: "related")
            
        Returns:
            Success status
        """
        if key1 not in self._memory or key2 not in self._memory:
            return False
        
        # Initialize relationships for keys if not exists
        if key1 not in self._relationships:
            self._relationships[key1] = {}
        if key2 not in self._relationships:
            self._relationships[key2] = {}
        
        # Add bidirectional relationship
        self._relationships[key1][key2] = relationship_type
        self._relationships[key2][key1] = self._get_inverse_relationship(relationship_type)
        
        return True
    
    async def get_related_items(self, key: str) -> Dict[str, str]:
        """
        Get all items related to a specific item.
        
        Args:
            key: The key to find relationships for
            
        Returns:
            Dictionary of related item keys and relationship types
        """
        if key not in self._relationships:
            return {}
        
        return self._relationships[key]
    
    async def set_focus(self, keys: List[str]) -> bool:
        """
        Explicitly set the focus to specific items.
        
        Args:
            keys: List of keys to focus on
            
        Returns:
            Success status
        """
        # Validate keys
        valid_keys = [key for key in keys if key in self._memory]
        
        # Clear current focus
        self._attention_focus.clear()
        
        # Set new focus
        for key in valid_keys:
            self._attention_focus.add(key)
            # Increase attention scores for focused items
            self._memory[key]["metadata"]["attention_score"] = min(
                10, self._memory[key]["metadata"].get("attention_score", 5) + 2
            )
        
        # If focus is too large, prune it
        if len(self._attention_focus) > self.capacity / 2:
            self._prune_focus()
        
        return len(valid_keys) > 0
    
    async def get_focus(self) -> List[Dict[str, Any]]:
        """
        Get the currently focused items.
        
        Returns:
            List of focused items with their values and metadata
        """
        focused_items = []
        
        for key in self._attention_focus:
            if key in self._memory:
                focused_items.append({
                    "key": key,
                    "value": self._memory[key]["value"],
                    "metadata": self._memory[key]["metadata"]
                })
        
        # Sort by attention score (highest first)
        focused_items.sort(key=lambda x: -x["metadata"].get("attention_score", 0))
        
        return focused_items
        
    async def get_all_keys(self) -> List[str]:
        """
        Get all keys currently in working memory.
        
        Returns:
            List of all keys in the memory
        """
        return list(self._memory.keys())
    
    async def chunk_information(self, key: str, chunks: List[Dict[str, Any]]) -> bool:
        """
        Break down a complex item into smaller, related chunks.
        
        Args:
            key: The key of the item to chunk
            chunks: List of chunk dictionaries with 'key', 'value', and optional 'metadata'
            
        Returns:
            Success status
        """
        if key not in self._memory:
            return False
        
        # Get original item
        original_item = self._memory[key]
        
        # Create chunks and add relationships
        for chunk in chunks:
            chunk_key = f"{key}_{chunk.get('key', 'chunk')}"
            chunk_value = chunk.get("value")
            chunk_metadata = chunk.get("metadata", {})
            
            # Copy relevant metadata from original
            default_metadata = {
                "parent_key": key,
                "is_chunk": True,
                "attention_score": original_item["metadata"].get("attention_score", 5),
                "chunk_type": chunk.get("type", "generic")
            }
            chunk_metadata = {**default_metadata, **chunk_metadata}
            
            # Add chunk to memory
            await self.add(chunk_key, chunk_value, chunk_metadata)
            
            # Add relationship
            await self.add_relationship(key, chunk_key, "parent_of")
        
        return True
    
    def _prune_by_attention(self) -> None:
        """
        Remove items with lowest attention scores to maintain capacity.
        """
        # Sort items by attention score (lowest first)
        items_by_attention = sorted(
            self._memory.items(),
            key=lambda x: (
                x[1]["metadata"].get("attention_score", 0),
                -x[1]["metadata"].get("timestamp", 0)  # Newer items preserved
            )
        )
        
        # Calculate how many items to remove
        remove_count = len(self._memory) - self.capacity
        
        # Remove the lowest attention items
        for i in range(remove_count):
            if i < len(items_by_attention):
                key, _ = items_by_attention[i]
                
                # Remove from memory
                if key in self._memory:
                    del self._memory[key]
                
                # Remove from focus
                if key in self._attention_focus:
                    self._attention_focus.remove(key)
                
                # Remove relationships
                if key in self._relationships:
                    del self._relationships[key]
                
                # Remove from other items' relationships
                for rel_key, related_items in self._relationships.items():
                    if key in related_items:
                        related_items.remove(key)
    
    def _prune_focus(self) -> None:
        """
        Reduce the focus set to maintain manageable size.
        """
        # Calculate target size (half of capacity)
        target_size = max(1, int(self.capacity / 2))
        
        # If focus is already small enough, do nothing
        if len(self._attention_focus) <= target_size:
            return
        
        # Get focused items sorted by attention score (lowest first)
        focus_items = []
        for key in self._attention_focus:
            if key in self._memory:
                focus_items.append({
                    "key": key,
                    "attention_score": self._memory[key]["metadata"].get("attention_score", 0),
                    "timestamp": self._memory[key]["metadata"].get("timestamp", 0)
                })
        
        focus_items.sort(key=lambda x: (x["attention_score"], -x["timestamp"]))
        
        # Calculate how many items to remove
        remove_count = len(self._attention_focus) - target_size
        
        # Remove lowest attention items from focus
        for i in range(remove_count):
            if i < len(focus_items):
                key = focus_items[i]["key"]
                if key in self._attention_focus:
                    self._attention_focus.remove(key)
                    
                    # Decrease attention score slightly
                    if key in self._memory:
                        self._memory[key]["metadata"]["attention_score"] = max(
                            1, self._memory[key]["metadata"].get("attention_score", 5) - 0.5
                        )
    
    def _get_inverse_relationship(self, relationship_type: str) -> str:
        """
        Get the inverse of a relationship type.
        
        Args:
            relationship_type: The relationship type to invert
            
        Returns:
            The inverse relationship type
        """
        # Define known inverses
        inverses = {
            "parent_of": "child_of",
            "child_of": "parent_of",
            "contains": "part_of",
            "part_of": "contains",
            "precedes": "follows",
            "follows": "precedes",
            "causes": "caused_by",
            "caused_by": "causes",
            "depends_on": "required_for",
            "required_for": "depends_on"
        }
        
        # Return inverse if known, otherwise use same type (symmetric relationship)
        return inverses.get(relationship_type, relationship_type)