# consumers.py
import json
from channels.generic.websocket import AsyncWebsocketConsumer

class RowUpdateConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        self.room_name = "row_updates"
        self.room_group_name = f"group_{self.room_name}"

        # Join the group
        await self.channel_layer.group_add(
            self.room_group_name,
            self.channel_name
        )
        await self.accept()

    async def disconnect(self, close_code):
        # Leave the group
        await self.channel_layer.group_discard(
            self.room_group_name,
            self.channel_name
        )

    # Receive message from WebSocket
    async def receive(self, text_data):
        text_data_json = json.loads(text_data)
        message = text_data_json['message']

        # Broadcast the message to the group
        await self.channel_layer.group_send(
            self.room_group_name,
            {
                'type': 'row_update_message',
                'message': message
            }
        )

    # Receive message from the group
    async def row_update_message(self, event):
        message = event['message']
        print("hi")
        # Send message to WebSocket
        await self.send(text_data=json.dumps({
            
            'message': message
        }))
        
    async def row_update_message1(self, event):
        message = event["message"]
        await self.send(text_data=json.dumps({
            "message": message
        }))
# fetchimage/consumers.py
