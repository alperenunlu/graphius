from PIL import Image
from tempfile import TemporaryDirectory


class Visual:
    def __init__(self, gif_path):
        self.gif_path = gif_path
        self.imgname = 0
        self.tempdir = TemporaryDirectory()
        self.states = []

    def add_state(self, graph, label=None):
        self.states.append(
            graph.visualize(label).render(
                filename=f"{self.tempdir.name}/{self.imgname}",
                format="png",
                cleanup=True,
            )
        )
        self.imgname += 1

    def save_gif(self):
        images = []
        max_width = 0
        max_height = 0
        for state in self.states:
            img = Image.open(state)
            max_width = max(max_width, img.width)
            max_height = max(max_height, img.height)
            images.append(img)

        for i in range(len(images)):
            images[i] = images[i].resize((max_width, max_height))

        images[0].save(
            self.gif_path,
            save_all=True,
            append_images=images[1:],
            optimize=False,
            duration=1000,
            loop=0,
        )

        return self.gif_path
