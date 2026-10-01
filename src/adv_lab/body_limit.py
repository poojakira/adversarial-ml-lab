"""Bound request bytes before framework parsing, including chunked transfers."""

from starlette.responses import JSONResponse


class RequestBodyLimit:
    def __init__(self, app, max_bytes: int):
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return None
        messages = []
        total = 0
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return None
            total += len(message.get("body", b""))
            if total > self.max_bytes:
                response = JSONResponse({"detail": "Request body too large"}, status_code=413)
                await response(scope, receive, send)
                return None
            messages.append(message)
            if not message.get("more_body", False):
                break
        index = 0

        async def replay():
            nonlocal index
            if index < len(messages):
                message = messages[index]
                index += 1
                return message
            return await receive()

        await self.app(scope, replay, send)
        return None
