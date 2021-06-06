import discord
from io import open

llave = open("llave.txt", "r").read()


cliente = discord.Client()
@cliente.event
async def on_message(mensaje):
    if mensaje.content.find("!hola-mundo")!=-1:
        await mensaje.channel.send("hola! desde discord")
cliente.run(llave)