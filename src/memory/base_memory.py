"""
Base Memory Module

This module provides the BaseMemory class which is the foundation for all memory types
in the advanced autonomous research system.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional


class BaseMemory(ABC):
    """
    Abstract base class for all memory types in the research system.
    
    This class defines the common interface that all memory types must implement.
    """
    
    def __init__(self, name: str):
        """
        Initialize a BaseMemory instance.
        
        Args:
            name: The name/identifier for this memory system
        """
        self.name = name
    
    @abstractmethod
    async def add(self, key: str, value: Any, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Add an item to memory.
        
        Args:
            key: The key to store the value under
            value: The value to store
            metadata: Optional metadata about the value
            
        Returns:
            Success status
        """
        pass
    
    @abstractmethod
    async def get(self, key: str) -> Optional[Any]:
        """
        Retrieve an item from memory by key.
        
        Args:
            key: The key to retrieve
            
        Returns:
            The stored value, or None if not found
        """
        pass
    
    @abstractmethod
    async def search(self, query: Any, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Search memory for relevant items.
        
        Args:
            query: The search query
            limit: Maximum number of results to return
            
        Returns:
            List of matching items with metadata
        """
        pass
    
    @abstractmethod
    async def update(self, key: str, value: Any, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Update an existing item in memory.
        
        Args:
            key: The key of the item to update
            value: The new value
            metadata: Optional new metadata
            
        Returns:
            Success status
        """
        pass
    
    @abstractmethod
    async def delete(self, key: str) -> bool:
        """
        Delete an item from memory.
        
        Args:
            key: The key of the item to delete
            
        Returns:
            Success status
        """
        pass
    
    @abstractmethod
    async def clear(self) -> bool:
        """
        Clear all items from memory.
        
        Returns:
            Success status
        """
        pass