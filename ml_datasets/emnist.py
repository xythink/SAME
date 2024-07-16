from torchvision.datasets import EMNIST as TVEMNIST


class EMNISTLetters(TVEMNIST):
    def __init__(self, root="./data/emnist", **kwargs):
        super().__init__(root, split="letters", **kwargs)
        # Images are transposed by default. Fix this.
        self.data = self.data.permute(0, 2, 1)

class EMNISTDigits(TVEMNIST):
    def __init__(self, root="./data/emnist", **kwargs):
        super().__init__(root, split="digits", **kwargs)
        # Images are transposed by default. Fix this.
        self.data = self.data.permute(0, 2, 1)
