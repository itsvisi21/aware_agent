import asyncio
from datetime import datetime
from typing import Dict, List, Any

from deploy.config import DeploymentConfig
from src.monitoring import MonitoringService


class MessageBatcher:
    def __init__(self, batch_size: int = 10, batch_interval: float = 0.1):
        self.config = DeploymentConfig()
        self.monitoring = MonitoringService()
        self.batch_size = batch_size
        self.batch_interval = batch_interval
        self.message_queue: Dict[str, List[Dict[str, Any]]] = {}
        self.processing = False
        self._lock = asyncio.Lock()

    async def add_message(self, client_id: str, message: Dict[str, Any]) -> None:
        """Add a message to the batch queue"""
        async with self._lock:
            if client_id not in self.message_queue:
                self.message_queue[client_id] = []
            self.message_queue[client_id].append(message)

            # Start processing if not already running
            if not self.processing:
                self.processing = True
                asyncio.create_task(self._process_batches())

    async def _process_batches(self) -> None:
        """Process message batches"""
        while True:
            async with self._lock:
                if not self.message_queue:
                    self.processing = False
                    return

                # Process each client's messages
                for client_id, messages in list(self.message_queue.items()):
                    if not messages:
                        continue

                    # Get batch of messages
                    batch = messages[:self.batch_size]
                    self.message_queue[client_id] = messages[self.batch_size:]

                    # Process batch
                    if batch:
                        await self._process_batch(client_id, batch)

                # Remove empty queues
                self.message_queue = {
                    k: v for k, v in self.message_queue.items()
                    if v
                }

            # Wait before next batch
            await asyncio.sleep(self.batch_interval)

    async def _process_batch(self, client_id: str, messages: List[Dict[str, Any]]) -> None:
        """Process a batch of messages"""
        try:
            # Add metadata
            batch_metadata = {
                'timestamp': datetime.now().isoformat(),
                'batch_size': len(messages),
                'client_id': client_id
            }

            # Process messages
            processed_messages = []
            for message in messages:
                # Add processing timestamp
                message['processed_at'] = datetime.now().isoformat()
                processed_messages.append(message)

            # Track metrics
            self.monitoring.track_messages_processed(len(processed_messages))

            # Return processed messages
            return {
                'metadata': batch_metadata,
                'messages': processed_messages
            }

        except Exception as e:
            self.monitoring.log_error(f"Batch processing error: {str(e)}")
            return None

    async def get_pending_messages(self, client_id: str) -> List[Dict[str, Any]]:
        """Get pending messages for a client"""
        async with self._lock:
            return self.message_queue.get(client_id, [])

    async def clear_messages(self, client_id: str) -> None:
        """Clear messages for a client"""
        async with self._lock:
            if client_id in self.message_queue:
                del self.message_queue[client_id]

    async def get_queue_size(self) -> Dict[str, int]:
        """Get current queue sizes"""
        async with self._lock:
            return {
                client_id: len(messages)
                for client_id, messages in self.message_queue.items()
            }
