class NoCompress:
    
    def __init__(*args, **kwargs):
        pass
    
    def compress(self, states):
        return states, None
    
    def decompress(self, states, shapes):
        return states