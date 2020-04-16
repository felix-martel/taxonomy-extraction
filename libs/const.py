from enum import Enum

class Prefix(Enum):
    OWL = "owl:"
    DBO = "dbo:"
    RDF = "rdf:"
    
    def short(self):
        return self.value.replace(":", "")
    
TOY_GRAPH = "data/demo.txt"
AUDIO_BEEP = "data/bip.mp3"


