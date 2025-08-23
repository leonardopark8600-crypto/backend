from fastapi import APIRouter, WebSocket, WebSocketDisconnect

router = APIRouter()
connections = set()

@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    connections.add(websocket)
    await broadcast_count()

    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        connections.remove(websocket)
        await broadcast_count()

async def broadcast_count():
    count = len(connections)
    for conn in connections:
        await conn.send_json({"count": count})