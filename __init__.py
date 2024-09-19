from .nodes.musicNode import MusicNode,AudioPlayNode,AudioToDictNode
from .nodes.StableAudioNode import StableAudioNode


NODE_CLASS_MAPPINGS = {
    "Musicgen_": MusicNode,
    "AudioPlay":AudioPlayNode,
    "AudioToDictNode":AudioToDictNode,
    "StableAudio_":StableAudioNode
}


NODE_DISPLAY_NAME_MAPPINGS = {
    "Musicgen_": "Music Gen",
    "AudioPlay":"Audio Play ♾️Sound Lab",
    "StableAudio_":"Stable Audio"
}

# web ui的节点功能issue
WEB_DIRECTORY = "./web"