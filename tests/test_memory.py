from multiagent_dev.agents.base import AgentMessage
from multiagent_dev.memory.memory import MemoryService


def test_memory_stores_messages_and_notes() -> None:
    memory = MemoryService()
    message = AgentMessage(sender="a", recipient="b", content="hello")

    memory.append_message("session-1", message)
    messages = memory.get_messages("session-1")

    assert messages == [message]

    memory.save_note("plan", "do the thing")
    assert memory.get_note("plan") == "do the thing"
