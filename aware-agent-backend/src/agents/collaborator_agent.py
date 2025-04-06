from typing import Dict, Any, List
from .base_agent import BaseAgent

class CollaboratorAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="Collaborator",
            role="Team Coordination"
        )
        self.team_members: Dict[str, Any] = {}
        self.tasks: Dict[str, Any] = {}
        self.progress_tracking: Dict[str, Any] = {}

    async def process_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a collaboration-related message and return a response.
        """
        content = message.get('content', '')
        self.add_to_history(message)

        # Update team and task information
        self._update_collaboration_info(content)

        # Generate collaboration response
        response = await self._generate_collaboration_response(content)

        # Update state with new information
        self.update_state({
            'last_action': content,
            'active_tasks': list(self.tasks.keys())
        })

        return {
            'id': message.get('id', '') + '_response',
            'content': response,
            'sender': 'agent',
            'timestamp': message.get('timestamp', 0),
            'metadata': {
                'agent': self.name,
                'role': self.role,
                'mode': 'collab',
                'team_members': self.team_members,
                'tasks': self.tasks,
                'progress': self.progress_tracking
            }
        }

    def _update_collaboration_info(self, content: str) -> None:
        """
        Update team members, tasks, and progress tracking based on message content.
        """
        # Simple team member tracking for demonstration
        if 'add team member' in content.lower():
            member = content.split('add team member')[-1].strip()
            if member and member not in self.team_members:
                self.team_members[member] = {
                    'role': 'team member',
                    'tasks_assigned': []
                }

        # Track tasks
        if 'task' in content.lower():
            task = content.split('task')[-1].strip()
            if task and task not in self.tasks:
                self.tasks[task] = {
                    'status': 'pending',
                    'assigned_to': None,
                    'progress': 0
                }

        # Update progress
        if 'progress' in content.lower():
            task = content.split('progress')[0].strip()
            if task in self.tasks:
                self.progress_tracking[task] = {
                    'last_update': content,
                    'status': 'in_progress'
                }

    async def _generate_collaboration_response(self, content: str) -> str:
        """
        Generate a collaboration-focused response based on the message content.
        """
        context = self.get_context()
        
        # Simple response generation for demonstration
        response = f"Based on your collaboration request about '{content}', "
        if self.team_members:
            response += f"Team members: {', '.join(self.team_members.keys())}. "
        if self.tasks:
            response += f"Active tasks: {', '.join(self.tasks.keys())}. "
        response += "Would you like me to help coordinate any specific aspect?"
        
        return response 