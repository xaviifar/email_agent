"""
Gmail Client Module

Async Gmail API client built on top of ``aiogoogle``.

Architecture note:
    Token refresh is handled upstream by ``GoogleAuthService.get_credentials``
    before this class is instantiated. The ``refresh_token`` is therefore
    intentionally *not* stored in ``user_creds`` to prevent ``aiogoogle`` from
    attempting its own internal refresh, which fails with ``invalid_grant``.
"""

import base64
import logging
from email.mime.text import MIMEText
from typing import Dict, List, Optional, Any, AsyncGenerator

import asyncio
from contextlib import asynccontextmanager

from aiogoogle import Aiogoogle
from google.oauth2.credentials import Credentials

logger = logging.getLogger(__name__)


class GmailAPIError(Exception):
    """Excepción base para errores de la API de Gmail."""
    pass


class GmailClient:
    """Async client for the Gmail REST API via ``aiogoogle``.

    All public methods are coroutines that open a short-lived ``Aiogoogle``
    session, execute the API call, and return a result or raise ``GmailAPIError``
    on failure.

    Attributes:
        user_creds: Aiogoogle user credential dict (access token + expiry).
        client_creds: Aiogoogle client credential dict (client ID/secret).
        user_email: Gmail user identifier; ``"me"`` refers to the
            authenticated user.
    """

    def __init__(self, credentials: Optional[Credentials] = None):
        """Initialise the client from a pre-refreshed ``Credentials`` object.

        Args:
            credentials: A ``google.oauth2.credentials.Credentials`` instance
                with a valid access token. When ``None``, the client is
                created in an unauthenticated state and ``is_authenticated``
                will return ``False``.
        """
        self.user_creds = None
        self.client_creds = None

        if credentials:
            self.user_creds = {
                "access_token": credentials.token,
                "expires_at": credentials.expiry.isoformat() if credentials.expiry else None,
            }
            self.client_creds = {
                "client_id": credentials.client_id,
                "client_secret": credentials.client_secret,
                "token_uri": credentials.token_uri or "https://oauth2.googleapis.com/token",
            }

        self.user_email = "me"

    def is_authenticated(self) -> bool:
        """Return ``True`` if the client holds a valid access token.

        Returns:
            ``True`` when ``user_creds`` is populated, ``False`` otherwise.
        """
        return self.user_creds is not None

    @asynccontextmanager
    async def _gmail_service(self) -> AsyncGenerator[Any, None]:
        """Async context manager that yields an active Aiogoogle Gmail service.

        Yields:
            A tuple of ``(aiogoogle_session, gmail_resource)`` ready for API
            calls.

        Raises:
            ValueError: If the client has not been authenticated.
        """
        if not self.user_creds:
            raise ValueError("Credenciales no configuradas.")

        async with Aiogoogle(user_creds=self.user_creds, client_creds=self.client_creds) as aiogoogle:
            gmail = await aiogoogle.discover("gmail", "v1")
            yield aiogoogle, gmail

    async def send(self, draft: Dict) -> str:
        """Send an email immediately.

        Args:
            draft: Dict with keys ``"to"``, ``"subject"``, ``"body"``, and
                optionally ``"cc"`` and ``"bcc"``.

        Returns:
            The ID of the sent message.

        Raises:
            GmailAPIError: If the API call fails.
        """
        try:
            async with self._gmail_service() as (aiogoogle, gmail):
                message = self._create_message(draft)
                sent_msg = await aiogoogle.as_user(
                    gmail.users.messages.send(userId=self.user_email, json=message)
                )
                return str(sent_msg.get('id'))
        except Exception as e:
            logger.exception(f"Error sending email: {e}")
            raise GmailAPIError(f"Error enviando correo: {e}") from e

    async def save_draft(self, draft: Dict) -> str:
        """Save an email as a Gmail draft.

        Args:
            draft: Dict with keys ``"to"``, ``"subject"``, ``"body"``, and
                optionally ``"cc"`` and ``"bcc"``.

        Returns:
            The ID of the created draft.

        Raises:
            GmailAPIError: If the API call fails.
        """
        try:
            async with self._gmail_service() as (aiogoogle, gmail):
                message = self._create_message(draft)
                body = {"message": message}
                draft_resp = await aiogoogle.as_user(
                    gmail.users.drafts.create(userId=self.user_email, json=body)
                )
                return str(draft_resp.get('id'))
        except Exception as e:
            logger.exception(f"save_draft FAILED: {type(e).__name__}: {e}")
            raise GmailAPIError(f"Error guardando borrador: {e}") from e

    async def update_draft(self, draft_id: str, draft: Dict) -> str:
        """Update an existing Gmail draft with new email content.

        Args:
            draft_id: Gmail draft ID.
            draft: Dict with keys ``"to"``, ``"subject"``, ``"body"``, and
                optionally ``"cc"`` and ``"bcc"``.

        Returns:
            The ID of the updated draft.

        Raises:
            GmailAPIError: If the API call fails.
        """
        try:
            async with self._gmail_service() as (aiogoogle, gmail):
                message = self._create_message(draft)
                body = {"id": draft_id, "message": message}
                updated_draft = await aiogoogle.as_user(
                    gmail.users.drafts.update(
                        userId=self.user_email,
                        id=draft_id,
                        json=body,
                    )
                )
                return str(updated_draft.get("id", draft_id))
        except Exception as e:
            logger.exception(f"Error updating draft {draft_id}: {e}")
            raise GmailAPIError(f"Error actualizando borrador: {e}") from e

    async def list_recent(self, count: int = 5) -> List[Dict]:
        """List the most recent inbox messages.

        Args:
            count: Maximum number of messages to return.

        Returns:
            A list of email summary dicts (``id``, ``from``, ``subject``,
            ``body`` snippet, ``labelIds``).

        Raises:
            GmailAPIError: If the API call fails.
        """
        try:
            async with self._gmail_service() as (aiogoogle, gmail):
                response = await aiogoogle.as_user(
                    gmail.users.messages.list(
                        userId=self.user_email,
                        maxResults=count,
                        q="label:INBOX",
                    )
                )

                messages = response.get("messages", [])
                if not messages:
                    return []

                tasks = [
                    self._get_message_details_async(aiogoogle, gmail, msg["id"], snippet_only=True)
                    for msg in messages
                ]
                return await asyncio.gather(*tasks)
        except Exception as e:
            logger.error(f"Error listing emails: {e}")
            raise GmailAPIError(f"No se pudieron listar los correos: {e}") from e

    async def search(self, query: str, max_results: int = 10) -> List[Dict]:
        """Search emails using Gmail query syntax.

        Automatically appends ``label:INBOX`` when the query does not already
        contain a ``label:`` or ``in:`` filter.

        Args:
            query: Gmail search query (e.g. ``"from:alice@example.com"``).
            max_results: Maximum number of results to return.

        Returns:
            A list of email summary dicts.

        Raises:
            GmailAPIError: If the API call fails.
        """
        try:
            async with self._gmail_service() as (aiogoogle, gmail):
                if "label:" in query.lower() or "in:" in query.lower():
                    final_query = query
                else:
                    final_query = f"{query} label:INBOX"

                response = await aiogoogle.as_user(
                    gmail.users.messages.list(
                        userId=self.user_email,
                        q=final_query,
                        maxResults=max_results,
                    )
                )

                messages = response.get("messages", [])
                if not messages:
                    return []

                tasks = [
                    self._get_message_details_async(aiogoogle, gmail, msg["id"], snippet_only=True)
                    for msg in messages
                ]
                results = await asyncio.gather(*tasks)
                return results
        except Exception as e:
            logger.error(f"Error searching emails: {e}")
            raise GmailAPIError(f"Error en la búsqueda: {e}") from e

    async def _get_message_details_async(
        self,
        aiogoogle,
        gmail_service,
        msg_id: str,
        snippet_only: bool = False,
    ) -> Dict:
        """Fetch details for a single message.

        Args:
            aiogoogle: Active ``Aiogoogle`` session.
            gmail_service: Discovered Gmail resource.
            msg_id: Gmail message ID.
            snippet_only: If ``True``, fetches metadata + snippet only
                (faster). If ``False``, fetches the full message body.

        Returns:
            Dict with keys ``id``, ``from``, ``subject``, ``body``, and
            ``labelIds``.

        Raises:
            GmailAPIError: If the API call fails.
        """
        format_ = "metadata" if snippet_only else "full"
        try:
            msg = await aiogoogle.as_user(
                gmail_service.users.messages.get(
                    userId=self.user_email,
                    id=msg_id,
                    format=format_,
                )
            )

            payload = msg.get("payload", {})
            headers = payload.get("headers", [])
            subject = next((h["value"] for h in headers if h["name"] == "Subject"), "Sin asunto")
            from_ = next((h["value"] for h in headers if h["name"] == "From"), "Desconocido")
            to_ = next((h["value"] for h in headers if h["name"] == "To"), "")
            reply_to = next((h["value"] for h in headers if h["name"].lower() == "reply-to"), "")
            message_id_header = next(
                (h["value"] for h in headers if h["name"].lower() in ("message-id", "message-id".lower())),
                "",
            )
            references = next((h["value"] for h in headers if h["name"].lower() == "references"), "")
            body = msg.get("snippet", "")

            if not snippet_only:
                extracted = self._extract_body(payload)
                if extracted:
                    body = extracted

            return {
                "id": msg_id,
                "threadId": msg.get("threadId", ""),
                "from": from_,
                "to": to_,
                "reply_to": reply_to,
                "subject": subject,
                "body": body,
                "message_id_header": message_id_header,
                "references": references,
                "labelIds": msg.get("labelIds", []),
            }
        except Exception as e:
            logger.error(f"Error getting message details for {msg_id}: {e}")
            raise GmailAPIError(f"No se pudo obtener detalles del mensaje {msg_id}: {e}") from e

    async def get_message(self, msg_id: str) -> Dict:
        """Fetch full details for a single message by ID.

        Args:
            msg_id: Gmail message ID.

        Returns:
            Full email dict.

        Raises:
            GmailAPIError: If the API call fails.
        """
        try:
            async with self._gmail_service() as (aiogoogle, gmail):
                return await self._get_message_details_async(aiogoogle, gmail, msg_id, snippet_only=False)
        except Exception as e:
            logger.error(f"Error getting message {msg_id}: {e}")
            raise GmailAPIError(f"No se pudo obtener el mensaje: {e}") from e

    def _extract_body(self, payload: Dict) -> str:
        """Recursively extract plain-text body from a Gmail message payload.

        Args:
            payload: The ``payload`` field from a Gmail message resource.

        Returns:
            Decoded plain-text body string, or an empty string if not found.
        """
        body = ""
        if "parts" in payload:
            for part in payload["parts"]:
                if part.get("mimeType") == "text/plain":
                    data = part["body"].get("data")
                    if data:
                        body += base64.urlsafe_b64decode(data).decode()
                elif part.get("parts"):
                    body += self._extract_body(part)
        else:
            data = payload.get("body", {}).get("data")
            if data:
                body += base64.urlsafe_b64decode(data).decode()
        return body

    async def delete(self, msg_id: str) -> None:
        """Move a single email to the Trash.

        Args:
            msg_id: Gmail message ID, or a dict containing an ``"id"`` key.

        Raises:
            GmailAPIError: If the API call fails.
        """
        if isinstance(msg_id, dict):
            msg_id = msg_id.get("id")

        try:
            async with self._gmail_service() as (aiogoogle, gmail):
                await aiogoogle.as_user(
                    gmail.users.messages.trash(userId=self.user_email, id=msg_id)
                )
        except Exception as e:
            logger.error(f"Error deleting email {msg_id}: {e}")
            raise GmailAPIError(f"Error eliminando correo: {e}") from e

    async def delete_batch(self, ids: List[str]) -> None:
        """Move multiple emails to the Trash in a single API call.

        Args:
            ids: List of Gmail message IDs to trash.

        Raises:
            GmailAPIError: If the API call fails.
        """
        try:
            async with self._gmail_service() as (aiogoogle, gmail):
                await aiogoogle.as_user(
                    gmail.users.messages.batchModify(
                        userId=self.user_email,
                        json={"ids": ids, "addLabelIds": ["TRASH"]},
                    )
                )
        except Exception as e:
            logger.error(f"Error deleting batch: {e}")
            raise GmailAPIError(f"Error eliminando correos: {e}") from e

    async def mark_read(self, email_obj: Dict) -> None:
        """Mark a single email as read.

        Args:
            email_obj: Email dict containing at least an ``"id"`` key.

        Raises:
            GmailAPIError: If the API call fails.
        """
        await self.mark_read_batch([email_obj["id"]])

    async def mark_read_batch(self, ids: List[str]) -> None:
        """Mark multiple emails as read.

        Args:
            ids: List of Gmail message IDs.

        Raises:
            GmailAPIError: If the API call fails.
        """
        try:
            async with self._gmail_service() as (aiogoogle, gmail):
                await aiogoogle.as_user(
                    gmail.users.messages.batchModify(
                        userId=self.user_email,
                        json={"ids": ids, "removeLabelIds": ["UNREAD"]},
                    )
                )
        except Exception as e:
            logger.error(f"Error marking read batch: {e}")
            raise GmailAPIError(f"Error marcando como leídos: {e}") from e

    async def mark_unread(self, email_obj: Dict) -> None:
        """Mark a single email as unread.

        Args:
            email_obj: Email dict containing at least an ``"id"`` key.

        Raises:
            GmailAPIError: If the API call fails.
        """
        await self.mark_unread_batch([email_obj["id"]])

    async def mark_unread_batch(self, ids: List[str]) -> None:
        """Mark multiple emails as unread.

        Args:
            ids: List of Gmail message IDs.

        Raises:
            GmailAPIError: If the API call fails.
        """
        try:
            async with self._gmail_service() as (aiogoogle, gmail):
                await aiogoogle.as_user(
                    gmail.users.messages.batchModify(
                        userId=self.user_email,
                        json={"ids": ids, "addLabelIds": ["UNREAD"]},
                    )
                )
        except Exception as e:
            logger.error(f"Error marking unread batch: {e}")
            raise GmailAPIError(f"Error marcando como no leídos: {e}") from e

    async def modify_labels(
        self,
        ids: List[str],
        add_labels: List[str] = None,
        remove_labels: List[str] = None,
    ) -> None:
        """Add and/or remove labels from multiple emails in one API call.

        Args:
            ids: List of Gmail message IDs to modify.
            add_labels: Label IDs to add (e.g. ``["IMPORTANT"]``).
            remove_labels: Label IDs to remove (e.g. ``["INBOX"]``).

        Raises:
            GmailAPIError: If the API call fails.
        """
        try:
            async with self._gmail_service() as (aiogoogle, gmail):
                json_data: Dict[str, Any] = {"ids": ids}
                if add_labels:
                    json_data["addLabelIds"] = add_labels
                if remove_labels:
                    json_data["removeLabelIds"] = remove_labels

                await aiogoogle.as_user(
                    gmail.users.messages.batchModify(userId=self.user_email, json=json_data)
                )
        except Exception as e:
            logger.error(f"Error modifying labels: {e}")
            raise GmailAPIError(f"Error modificando etiquetas: {e}") from e

    async def get_label_details(self, label_id: str = "INBOX") -> Dict:
        """Retrieve message and thread counts for a Gmail label.

        Args:
            label_id: Gmail label ID (e.g. ``"INBOX"``, ``"UNREAD"``).

        Returns:
            Dict with keys ``name``, ``messagesTotal``, ``messagesUnread``,
            ``threadsTotal``, and ``threadsUnread``.

        Raises:
            GmailAPIError: If the API call fails.
        """
        try:
            async with self._gmail_service() as (aiogoogle, gmail):
                label = await aiogoogle.as_user(
                    gmail.users.labels.get(userId=self.user_email, id=label_id)
                )
                return {
                    "name": label.get("name", ""),
                    "messagesTotal": label.get("messagesTotal", 0),
                    "messagesUnread": label.get("messagesUnread", 0),
                    "threadsTotal": label.get("threadsTotal", 0),
                    "threadsUnread": label.get("threadsUnread", 0),
                }
        except Exception as e:
            logger.error(f"Error getting label details: {e}")
            raise GmailAPIError(f"No se pudo obtener información de la etiqueta: {e}") from e

    async def list_labels(self) -> List[Dict]:
        """Return all labels for the authenticated Gmail account."""
        try:
            async with self._gmail_service() as (aiogoogle, gmail):
                response = await aiogoogle.as_user(
                    gmail.users.labels.list(userId=self.user_email)
                )
                return response.get("labels", []) or []
        except Exception as e:
            logger.error(f"Error listing labels: {e}")
            raise GmailAPIError(f"No se pudieron listar las etiquetas: {e}") from e

    def _create_message(self, draft: Dict) -> Dict:
        """Build a base64-encoded MIME message for the Gmail API.

        Args:
            draft: Dict with keys ``"to"``, ``"subject"``, ``"body"``, and
                optionally ``"cc"`` and ``"bcc"``.

        Returns:
            A dict with a single ``"raw"`` key containing the base64url-encoded
            MIME message, ready to pass as the JSON body to the Gmail API.
        """
        message = MIMEText(draft.get("body", ""))
        message["to"] = draft.get("to", "")
        message["subject"] = draft.get("subject", "Sin asunto")

        if draft.get("cc"):
            cc = draft["cc"]
            message["Cc"] = ", ".join(cc) if isinstance(cc, list) else cc

        if draft.get("bcc"):
            bcc = draft["bcc"]
            message["Bcc"] = ", ".join(bcc) if isinstance(bcc, list) else bcc

        if draft.get("in_reply_to"):
            message["In-Reply-To"] = draft["in_reply_to"]

        if draft.get("references"):
            message["References"] = draft["references"]

        raw = base64.urlsafe_b64encode(message.as_bytes()).decode()
        payload = {"raw": raw}
        if draft.get("thread_id"):
            payload["threadId"] = draft["thread_id"]
        return payload

    async def list_drafts(self, count: int = 10) -> List[Dict]:
        """List existing Gmail drafts with subject and recipient details.

        Args:
            count: Maximum number of drafts to return.

        Returns:
            A list of dicts with keys ``id``, ``message_id``, ``subject``,
            and ``to``.

        Raises:
            GmailAPIError: If the API call fails.
        """
        try:
            async with self._gmail_service() as (aiogoogle, gmail):
                response = await aiogoogle.as_user(
                    gmail.users.drafts.list(userId=self.user_email, maxResults=count)
                )

                drafts = response.get("drafts", [])
                if not drafts:
                    return []

                results = []
                for draft in drafts:
                    d_id = draft["id"]
                    detail = await aiogoogle.as_user(
                        gmail.users.drafts.get(userId=self.user_email, id=d_id, format="full")
                    )
                    msg = detail.get("message", {})
                    payload = msg.get("payload", {})
                    headers = payload.get("headers", [])

                    subject = next((h["value"] for h in headers if h["name"] == "Subject"), "(Sin asunto)")
                    to = next((h["value"] for h in headers if h["name"] == "To"), "(Sin destinatario)")

                    results.append({
                        "id": d_id,
                        "message_id": msg.get("id"),
                        "subject": subject,
                        "to": to,
                    })

                return results
        except Exception as e:
            logger.error(f"Error listing drafts: {e}")
            raise GmailAPIError(f"Error listando borradores: {e}") from e

    async def send_draft(self, draft_id: str) -> str:
        """Send an existing Gmail draft by its draft ID.

        Args:
            draft_id: The Gmail draft ID (not the message ID).

        Returns:
            The ID of the sent message.

        Raises:
            GmailAPIError: If the API call fails.
        """
        try:
            async with self._gmail_service() as (aiogoogle, gmail):
                sent_msg = await aiogoogle.as_user(
                    gmail.users.drafts.send(
                        userId=self.user_email,
                        json={"id": draft_id},
                    )
                )
                return str(sent_msg.get('id'))
        except Exception as e:
            logger.error(f"Error sending draft: {e}")
            raise GmailAPIError(f"Error enviando borrador: {e}") from e
        
    async def create_labels(self, labels: List):

        try:
            async with self._gmail_service() as (aiogoogle, gmail):
                for label in labels:
                    create_payload = {
                        "name": label.get("name"),
                        "labelListVisibility": label.get("labelListVisibility", "labelShow"),
                        "messageListVisibility": label.get("messageListVisibility", "show"),
                    }

                    color = label.get("color") if isinstance(label, dict) else None
                    text_color = None
                    background_color = None
                    if isinstance(color, dict):
                        text_color = color.get("textColor") or color.get("text")
                        background_color = color.get("backgroundColor") or color.get("background")

                    response = await aiogoogle.as_user(
                        gmail.users.labels.create(
                            userId=self.user_email,
                            json=create_payload
                        )
                    )

                    label_id = response["id"]

                    if text_color and background_color:
                        try:
                            await aiogoogle.as_user(
                                gmail.users.labels.patch(
                                    userId=self.user_email,
                                    id=label_id,
                                    json={
                                        "color": {
                                            "textColor": text_color,
                                            "backgroundColor": background_color
                                        }
                                    }
                                )
                            )
                        except Exception as color_error:
                            logger.warning(
                                "No se pudo aplicar el color a la etiqueta; la etiqueta se mantiene creada sin color. "
                                f"label={create_payload.get('name')} text_color={text_color} "
                                f"background_color={background_color} error={color_error}"
                            )

        except Exception as e:
            logger.error(f"Error creating labels: {e}")
            raise GmailAPIError(f"Error creando las etiquetas: {e}") from e

    async def update_label(self, label_id: str, message_id: str):
        try:

            async with self._gmail_service() as (aiogoogle, gmail):
                response = await aiogoogle.as_user(
                    gmail.messages.modify(
                        userId=self.user_email,
                        id=message_id,
                        json={
                            "addLabelIds": [label_id]
                        }
                    )
                )
                return response
        
        except Exception as e: 
            logger.error(f"Error adding label: {e}")
            raise GmailAPIError(f"Error etiquetando el email: {e}")
        