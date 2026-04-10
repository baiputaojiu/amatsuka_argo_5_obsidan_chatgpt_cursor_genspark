class UILogSink:
    def __init__(self, cb): self.cb=cb
    def info(self, msg):
        if self.cb: self.cb(msg)
