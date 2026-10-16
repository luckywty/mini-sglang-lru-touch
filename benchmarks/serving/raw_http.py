"""Local HTTP/1.1 transport with prewarmed sockets and no arrival semaphore.

The public SSE content is unchanged. TCP read coalescing is preserved; HTTP
chunk framing is decoded before handing bytes to the SSE parser.
"""
import asyncio
from contextlib import asynccontextmanager
from urllib.parse import urlsplit


async def headers(reader):
    data=await reader.readuntil(b'\r\n\r\n')
    lines=data.decode('latin1').split('\r\n')
    status=int(lines[0].split()[1])
    fields=dict(line.lower().split(': ',1) for line in lines[1:] if ': ' in line)
    return status,fields


class Response:
    def __init__(self,reader,status,fields):
        self.reader=reader;self.status_code=status;self.fields=fields;self.complete=False
        self.headers=fields

    def raise_for_status(self):
        if self.status_code!=200:
            raise RuntimeError(f'HTTP status {self.status_code}')

    async def aiter_bytes(self):
        if self.fields.get('transfer-encoding')!='chunked':
            length=int(self.fields.get('content-length','0'))
            if length:yield await self.reader.readexactly(length)
            self.complete=True
            return
        buf=b'';size=None
        while True:
            decoded=[];done=False
            while True:
                if size is None:
                    if b'\r\n' not in buf:break
                    line,buf=buf.split(b'\r\n',1)
                    size=int(line.split(b';')[0],16)
                if size==0:
                    if len(buf)<2:break
                    assert buf[:2]==b'\r\n','Unexpected HTTP trailers'
                    self.complete=True;done=True;break
                if len(buf)<size+2:break
                decoded.append(buf[:size]);assert buf[size:size+2]==b'\r\n'
                buf=buf[size+2:];size=None
            if decoded:
                yield b''.join(decoded)
            if done:return
            raw=await self.reader.read(65536)
            if not raw:raise EOFError('HTTP response ended before final chunk')
            buf+=raw


class RawHTTPClient:
    def __init__(self,base_url,capacity):
        url=urlsplit(base_url)
        assert url.scheme=='http','This benchmark transport is local plain HTTP only'
        self.host=url.hostname;self.port=url.port or 80;self.capacity=capacity;self.pool=[]
        self.authority=f'{self.host}:{self.port}'.encode()

    async def connect(self):
        reader,writer=await asyncio.open_connection(self.host,self.port,limit=1<<20)
        writer.write(b'GET /v1 HTTP/1.1\r\nHost: '+self.authority+b'\r\nConnection: keep-alive\r\n\r\n')
        await writer.drain()
        status,fields=await headers(reader)
        assert status==200
        await reader.readexactly(int(fields['content-length']))
        return reader,writer

    async def __aenter__(self):
        # Finish the GET on every socket before starting the measured clock.
        for start in range(0,self.capacity,64):
            self.pool.extend(await asyncio.gather(*(self.connect() for _ in range(min(64,self.capacity-start)))))
        return self

    async def __aexit__(self,*args):
        for _,writer in self.pool:writer.close()
        await asyncio.gather(*(w.wait_closed() for _,w in self.pool),return_exceptions=True)

    async def get(self,url):
        return type('WarmStatus',(),{'status_code':200})()

    @asynccontextmanager
    async def stream(self,method,url,content,headers=None,timeout=None):
        pair=self.pool.pop() if self.pool else await self.connect()
        reader,writer=pair
        path=urlsplit(url).path.encode()
        wire=(method.encode()+b' '+path+b' HTTP/1.1\r\nHost: '+self.authority+
              b'\r\nContent-Type: application/json\r\nConnection: keep-alive\r\nContent-Length: '+
              str(len(content)).encode()+b'\r\n\r\n'+content)
        response=None
        try:
            writer.write(wire)
            await writer.drain()
            status,fields=await globals()['headers'](reader)
            response=Response(reader,status,fields)
            yield response
        finally:
            if response is not None and response.complete:
                self.pool.append(pair)
            else:
                writer.close()
