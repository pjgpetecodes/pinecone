from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Optional, Dict, Any


@dataclass
class LogEvent:
    timestamp: datetime
    level: str
    service: str
    message: str
    trace_id: Optional[str] = None
    user_id: Optional[str] = None
    attributes: Optional[Dict[str, Any]] = None

    def to_text(self) -> str:
        parts = [
            f"level={self.level}",
            f"service={self.service}",
            f"message={self.message}",
        ]
        if self.user_id:
            parts.append(f"user={self.user_id}")
        
        # Include key attributes in text for better embedding representation
        if self.attributes:
            if self.attributes.get("country"):
                parts.append(f"country={self.attributes.get('country')}")
            if self.attributes.get("category"):
                parts.append(f"category={self.attributes.get('category')}")
            if self.attributes.get("product"):
                parts.append(f"product={self.attributes.get('product')}")
            if self.attributes.get("amount"):
                parts.append(f"amount=${self.attributes.get('amount')}")
        
        return " | ".join(parts)

    def to_metadata(self) -> Dict[str, Any]:
        meta = {
            "timestamp": self.timestamp.isoformat(),
            "level": self.level,
            "service": self.service,
            "message": self.message,
        }
        if self.trace_id:
            meta["trace_id"] = self.trace_id
        if self.user_id:
            meta["user_id"] = self.user_id
        # Flatten attributes into top-level metadata (Pinecone doesn't support nested dicts)
        if self.attributes:
            for k, v in self.attributes.items():
                meta[f"attr_{k}"] = v
        return meta
