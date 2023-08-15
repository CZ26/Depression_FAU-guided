class Sample:

    def __init__(self, vid, speaker, label, CNN_resnet, audio, au_spec):
        self.vid = vid
        self.speaker = speaker
        self.label = label
        self.audio = audio
        self.CNN_resnet = CNN_resnet
        self.au_spec = au_spec
