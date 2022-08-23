class Buffer(object):
    def __init__(self, maxSize):
        self.maxSize    = maxSize
        self.candidates = []

    def refresh(self, candidates):
        pass
