from langchain.tools import tool
from external.gmail_client import GmailClient


class GmailTools:
    def __init__(self, gmail_client: GmailClient):
        self.gmail_client = gmail_client

        @tool(description="Gets emails from the inbox")
        async def read_inbox(query: str = "", limit: int = 10) -> list[dict]:
            if query:
                return await self.gmail_client.search(
                    query,
                    max_results=limit
                )

            return await self.gmail_client.list_recent(
                count=limit
            )

        @tool(description="Deletes emails by ID in batch")
        async def delete_batch(messages: list[str]) -> None:
            await self.gmail_client.delete_batch(
                ids=messages
            )

        @tool(description="Searches emails")
        async def search_emails(query: str = "", limit: int = 10) -> list[dict]:
            return await self.gmail_client.search(
                query,
                max_results=limit
            )

        @tool(description="Delete a unique email by ID")
        async def delete_email(email_id: str):
            await self.gmail_client.delete(email_id)

        self.read_inbox = read_inbox
        self.delete_batch = delete_batch
        self.search_emails = search_emails
        self.delete_email = delete_email