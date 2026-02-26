"""WebSocket gesture detection endpoint."""
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, status, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Dict
import json
import httpx
from app.db.session import get_db
from app.core.config import settings
from app.core.security import decode_token

router = APIRouter(tags=["websocket"])

# Track active WebSocket connections
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}

    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections[client_id] = websocket

    def disconnect(self, client_id: str):
        if client_id in self.active_connections:
            del self.active_connections[client_id]

    async def broadcast(self, message: str, exclude_client: str = None):
        for client_id, connection in self.active_connections.items():
            if exclude_client and client_id == exclude_client:
                continue
            try:
                await connection.send_text(message)
            except:
                pass


manager = ConnectionManager()


@router.websocket("/api/v1/ws/gesture/{token}")
async def websocket_gesture_endpoint(
    websocket: WebSocket,
    token: str,
    db: AsyncSession = Depends(get_db)
):
    """WebSocket endpoint for real-time gesture detection."""
    # Validate token
    payload = decode_token(token)
    user_id = payload.get("sub")

    if not user_id:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return

    client_id = f"user_{user_id}"
    await manager.connect(websocket, client_id)

    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)

            # Process gesture if image data is provided
            if "image_data" in message and "lesson_id" in message:
                try:
                    # Send to ML service for detection
                    async with httpx.AsyncClient() as client:
                        ml_response = await client.post(
                            f"{settings.ml_service_url}{settings.ml_service_predict_endpoint}",
                            json={
                                "image_data": message["image_data"]
                            },
                            timeout=30.0
                        )

                        if ml_response.status_code == 200:
                            prediction = ml_response.json()

                            response = {
                                "status": "success",
                                "detected_sign": prediction.get("predicted_class"),
                                "confidence": prediction.get("confidence", 0),
                                "expected_sign": message.get("expected_sign"),
                                "correct": prediction.get("predicted_class") == message.get("expected_sign")
                            }
                        else:
                            response = {
                                "status": "error",
                                "message": "ML service error"
                            }
                except Exception as e:
                    response = {
                        "status": "error",
                        "message": str(e)
                    }

                await websocket.send_json(response)

            # Echo other messages
            elif "action" in message:
                await websocket.send_json({
                    "status": "received",
                    "action": message["action"]
                })

    except WebSocketDisconnect:
        manager.disconnect(client_id)
    except Exception as e:
        manager.disconnect(client_id)
        await websocket.close(code=status.WS_1011_SERVER_ERROR)
