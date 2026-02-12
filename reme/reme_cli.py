"""ReMe File System"""

import asyncio
import sys
from typing import AsyncGenerator

from prompt_toolkit import PromptSession

from .agent.chat import FsCli
from .core.enumeration import ChunkEnum
from .core.schema import StreamChunk
from .core.utils import execute_stream_task
from .reme_fs import ReMeFs
from .tool.fs import (
    BashTool,
    EditTool,
    FsMemorySearch,
    LsTool,
    ReadTool,
    WriteTool,
)
from .tool.gallery import ExecuteCode
from .tool.search import DashscopeSearch


class ReMeCli(ReMeFs):
    """ReMe Cli"""

    def __init__(self, *args, **kwargs):
        """Initialize ReMe with config."""
        super().__init__(*args, **kwargs)
        self.commands = {
            "/new": "Create a new conversation.",
            "/compact": "Compact messages into a summary.",
            "/exit": "Exit the application.",
            "/clear": "Clear the history.",
            "/help": "Show help.",
        }

    async def chat_with_remy(self, tool_result_max_size: int = 100, language: str = "zh", **kwargs):
        """Interactive CLI chat with Remy using simple streaming output."""
        fs_cli = FsCli(
            tools=[
                FsMemorySearch(
                    vector_weight=self.vector_weight,
                    candidate_multiplier=self.candidate_multiplier,
                ),
                BashTool(cwd=self.working_dir),
                LsTool(cwd=self.working_dir),
                ReadTool(cwd=self.working_dir),
                EditTool(cwd=self.working_dir),
                WriteTool(cwd=self.working_dir),
                ExecuteCode(),
                DashscopeSearch(),
            ],
            context_window_tokens=self.context_window_tokens,
            reserve_tokens=self.reserve_tokens,
            keep_recent_tokens=self.keep_recent_tokens,
            working_dir=self.working_dir,
            language=language,
            **kwargs,
        )
        session = PromptSession()

        # Print welcome banner
        print("\n========================================")
        print("  Welcome to Remy Chat!")
        print("========================================\n")

        async def chat(q: str) -> AsyncGenerator[StreamChunk, None]:
            """Execute chat query and yield streaming chunks."""
            stream_queue = asyncio.Queue()
            task = asyncio.create_task(
                fs_cli.call(
                    query=q,
                    stream_queue=stream_queue,
                    service_context=self.service_context,
                ),
            )
            async for _chunk in execute_stream_task(
                stream_queue=stream_queue,
                task=task,
                task_name="cli",
                output_format="chunk",
            ):
                yield _chunk

        while True:
            try:
                # Get user input (async)
                user_input = await session.prompt_async("You: ")
                user_input = user_input.strip()
                if not user_input:
                    continue

                # Handle commands
                if user_input == "/exit":
                    break

                if user_input == "/new":
                    result = await fs_cli.reset()
                    print(f"{result}\nConversation reset\n")
                    continue

                if user_input == "/compact":
                    result = await fs_cli.compact(force_compact=True)
                    print(f"{result}\nHistory compacted.\n")
                    continue

                if user_input == "/clear":
                    fs_cli.messages.clear()
                    print("History cleared.\n")
                    continue

                if user_input == "/help":
                    print("\nCommands:")
                    for command, description in self.commands.items():
                        print(f"  {command}: {description}")
                    continue

                # Stream processing state
                in_thinking = False
                in_answer = False

                try:
                    async for chunk in chat(user_input):
                        if chunk.chunk_type == ChunkEnum.THINK:
                            if not in_thinking:
                                print("\033[90mThinking: ", end="", flush=True)
                                in_thinking = True
                            print(chunk.chunk, end="", flush=True)

                        elif chunk.chunk_type == ChunkEnum.ANSWER:
                            if in_thinking:
                                print("\033[0m")  # reset color after thinking
                                in_thinking = False
                            if not in_answer:
                                print("\nRemy: ", end="", flush=True)
                                in_answer = True
                            print(chunk.chunk, end="", flush=True)

                        elif chunk.chunk_type == ChunkEnum.TOOL:
                            if in_thinking:
                                print("\033[0m")  # reset color after thinking
                                in_thinking = False
                            print(f"\033[36m  -> {chunk.chunk}\033[0m")

                        elif chunk.chunk_type == ChunkEnum.TOOL_RESULT:
                            tool_name = chunk.metadata.get("tool_name", "unknown")
                            result = chunk.chunk
                            if len(result) > tool_result_max_size:
                                result = result[:tool_result_max_size] + f"... ({len(chunk.chunk)} chars total)"
                            print(f"\033[36m  -> Tool result for {tool_name}: {result.strip()}\033[0m")

                        elif chunk.chunk_type == ChunkEnum.ERROR:
                            print(f"\n\033[91m[ERROR] {chunk.chunk}\033[0m")
                            # Also log the full error metadata if available
                            if chunk.metadata:
                                import traceback

                                traceback.print_exc()

                        elif chunk.chunk_type == ChunkEnum.DONE:
                            break

                except Exception as e:
                    print(f"\nStream error: {e}")

                # End current streaming line
                print("\n")
                print("----------------------------------------\n")

            except EOFError:
                break
            except KeyboardInterrupt:
                print("\nInterrupted.")
                break
            except Exception as e:
                print(f"Error: {e}")
                import traceback

                traceback.print_exc()

        print("\nGoodbye!\n")


async def async_main():
    """Main function for testing the ReMeFs CLI."""
    async with ReMeCli(*sys.argv[1:], log_to_console=False) as reme:
        await reme.chat_with_remy()


def main():
    """Main function for testing the ReMeFs CLI."""
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
