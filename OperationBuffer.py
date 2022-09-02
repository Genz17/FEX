class Buffer(object):
    def __init__(self, maxSize):
        self.maxSize    = maxSize
        self.bufferzone = []

    def refresh(self, candidate):
        change = False
        for inx, oldCandidate in enumerate(self.bufferzone):
            if oldCandidate.action == candidate.action and oldCandidate.error >= candidate.error:
                change = True
                changeinx = inx
                break
        if change:
            self.bufferzone.pop(changeinx)
        self.bufferzone.append(candidate)
        self.bufferzone = sorted(self.bufferzone, key=lambda x:x.error)

        if len(self.bufferzone) > self.maxSize:
            self.bufferzone.pop(-1)




