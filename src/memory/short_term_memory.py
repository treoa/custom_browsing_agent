"""
Short-Term Memory Module

This module provides the ShortTermMemory class which maintains immediate context
during research sessions.
"""

from typing import Dict, List, Any, Optional
import time
from collections import OrderedDict
import json

from .base_memory import BaseMemory


class ShortTermMemory(BaseMemory):
    """
    Short-Term Memory maintains immediate context during research sessions.
    
    This memory component stores recent information with automatic decay mechanisms
    and prioritization for critical information.
    """
    
    def __init__(
        self,
        name: str = "STM",
        capacity: int = 100,
        decay_time: float = 3600,  # 1 hour in seconds
    ):
        """
        Initialize a ShortTermMemory instance.
        
        Args:
            name: The name/identifier for this memory system (default: "STM")
            capacity: Maximum number of items to store (default: 100)
            decay_time: Time in seconds after which items are considered expired (default: 3600)
        """
        super().__init__(name=name)
        self.capacity = capacity
        self.decay_time = decay_time
        self._memory = OrderedDict()  # Using OrderedDict to maintain insertion order
    
    async def add(self, key: str, value: Any, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Add an item to short-term memory.
        
        Args:
            key: The key to store the value under
            value: The value to store
            metadata: Optional metadata about the value
            
        Returns:
            Success status
        """
        # Create metadata if none provided
        metadata = metadata or {}
        
        # Add timestamp and priority if not provided
        if "timestamp" not in metadata:
            metadata["timestamp"] = time.time()
        if "priority" not in metadata:
            metadata["priority"] = 1  # Default priority (1-10, higher is more important)
        
        # Create memory item
        memory_item = {
            "key": key,
            "value": value,
            "metadata": metadata
        }
        
        # Add to memory
        self._memory[key] = memory_item
        
        # Check if we're over capacity
        if len(self._memory) > self.capacity:
            # First try to remove expired items
            self._clear_expired()
            
            # If still over capacity, remove lowest priority items
            if len(self._memory) > self.capacity:
                self._prune_low_priority()
        
        return True
    
    async def get(self, key: str) -> Optional[Any]:
        """
        Retrieve an item from short-term memory by key.
        
        Args:
            key: The key to retrieve
            
        Returns:
            The stored value, or None if not found or expired
        """
        # Clear expired items first
        self._clear_expired()
        
        # Check if key exists
        if key not in self._memory:
            return None
        
        # Get the memory item
        memory_item = self._memory[key]
        
        # Update access timestamp to show this item is still relevant
        memory_item["metadata"]["last_accessed"] = time.time()
        
        # Update priority slightly to indicate usage
        if "priority" in memory_item["metadata"]:
            memory_item["metadata"]["priority"] = min(
                10, memory_item["metadata"]["priority"] + 0.1
            )
        
        # Move to end of OrderedDict to indicate recent use
        self._memory.move_to_end(key)
        
        return memory_item["value"]
    
    async def search(self, query: Any, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Search short-term memory for relevant items.
        
        Args:
            query: The search query (string or dict with field queries)
            limit: Maximum number of results to return
            
        Returns:
            List of matching items with metadata
        """
        # Clear expired items first
        self._clear_expired()
        
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
                            "metadata": item["metadata"]
                        })
                except:
                    # If serialization fails, just search the key
                    if query_lower in str(key).lower():
                        results.append({
                            "key": key,
                            "value": item["value"],
                            "metadata": item["metadata"]
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
                        "metadata": item["metadata"]
                    })
        
        # Sort results by priority (highest first) and timestamp (newest first)
        results.sort(key=lambda x: (
            -x["metadata"].get("priority", 0),
            -x["metadata"].get("timestamp", 0)
        ))
        
        # Limit results
        return results[:limit]
    
    async def update(self, key: str, value: Any, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Update an existing item in short-term memory.
        
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
        
        # Move to end of OrderedDict to indicate recent use
        self._memory.move_to_end(key)
        
        return True
    
    async def delete(self, key: str) -> bool:
        """
        Delete an item from short-term memory.
        
        Args:
            key: The key of the item to delete
            
        Returns:
            Success status
        """
        if key in self._memory:
            del self._memory[key]
            return True
        return False
    
    async def clear(self) -> bool:
        """
        Clear all items from short-term memory.
        
        Returns:
            Success status
        """
        self._memory.clear()
        return True
    
    def _clear_expired(self) -> None:
        """
        Remove expired items from memory.
        """
        current_time = time.time()
        expired_keys = []
        
        for key, item in self._memory.items():
            timestamp = item["metadata"].get("timestamp", 0)
            if current_time - timestamp > self.decay_time:
                expired_keys.append(key)
        
        for key in expired_keys:
            del self._memory[key]
    
    def _prune_low_priority(self) -> None:
        """
        Remove lowest priority items to maintain capacity.
        """
        # Sort items by priority (lowest first)
        items_by_priority = sorted(
            self._memory.items(),
            key=lambda x: x[1]["metadata"].get("priority", 0)
        )
        
        # Calculate how many items to remove
        remove_count = len(self._memory) - self.capacity
        
        # Remove the lowest priority items
        for i in range(remove_count):
            if i < len(items_by_priority):
                key, _ = items_by_priority[i]
                del self._memory[key]