import os

from PIL import Image


def is_image_url(url):
    if url.split("?")[0].split(".")[-1] in ["mp4", "avi", "mkv", "mov"]:
        return False
    return True


class ImageGridVisualizer:
    """
    A class to visualize a grid of images with descriptions. The visualizer generates an HTML file
    that displays images in a tabular format, with each row containing a description and a set of images.

    Author: Midhun Harikumar
    Email: midhun.harikumar@morphic.com

    Attributes:
    - filename: Path to the output HTML file.
    - rows: A list of rows where each row is a dictionary containing a description and a list of image filenames.
    - max_images: The maximum number of images in any row.
    - images_folder: Folder to store image files.
    - image_counter: A counter for generating unique image filenames.
    """

    def __init__(self, filename: str):
        """
        Initializes the ImageGridVisualizer.

        Parameters:
        - filename: The path to the output HTML file (should end with .html).
        """
        if not filename.endswith(".html"):
            filename = filename + ".html"
        self.filename = filename
        self.rows = []
        self.max_images = 0

        self.images_folder = os.path.splitext(self.filename)[0]
        if not os.path.exists(self.images_folder):
            os.makedirs(self.images_folder)

        self.image_counter = 0
        self.column_names = None

    def add_row(self, description: str, images: list[Image.Image | str | None]):
        """
        Adds a row to the image grid, which includes a description and a list of images.

        Parameters:
        - description: A string or HTML content that will appear in the first column.
        - images: A list of PIL Image objects or image file paths to be displayed in the row.
        """
        image_filenames = []
        for img in images:
            if img:
                if isinstance(img, Image.Image):
                    image_filename = f"image_{self.image_counter}.png"
                    img_path = os.path.join(self.images_folder, image_filename)
                    img.save(img_path, format="PNG")
                else:
                    image_filename = img
                image_filenames.append(image_filename)
                self.image_counter += 1
            else:
                image_filenames.append(None)

        self.rows.append({"description": description, "images": image_filenames})

        if len(images) > self.max_images:
            self.max_images = len(images)

    def add_row_video(self, description: str, video_urls: list[str]):
        self.image_counter += len(video_urls)
        self.rows.append({"description": description, "images": video_urls})

    def add_row_urls(self, description: str, image_urls: list[str]):
        assert sum(list(map(lambda x: x.startswith("gs"), image_urls))) == len(image_urls)
        image_filenames = [
            url.replace("gs://", "https://storage.cloud.google.com/") + "?authuser=0" for url in image_urls
        ]
        self.image_counter += len(image_urls)
        self.rows.append({"description": description, "images": image_filenames})
        self.max_images = max(self.max_images, len(image_urls))

    def add_set_column_names(self, column_names: list[str]):
        self.column_names = column_names

    def render(self):
        """
        Generates the HTML image grid and writes it to the specified output file.
        This method creates a table with descriptions in the first column and images in subsequent columns.
        The generated HTML file can be opened in a web browser to view the image grid.
        """
        html = '<table style="border-collapse: collapse; width: 100%;">\n'

        html += "<tr>\n"
        html += '<th style="border: 1px solid #dddddd; text-align: left; padding: 2px;">Description</th>\n'

        if self.column_names is None:
            self.column_names = [f"Image {i}" for i in range(1, self.max_images + 1)]

        for i in range(1, self.max_images + 1):
            html += f'<th style="border: 1px solid #dddddd; text-align: center; padding: 2px;">{self.column_names[i - 1]}</th>\n'
        html += "</tr>\n"

        for row in self.rows:
            html += '<tr style="height: 250px;">\n'

            description = row["description"]
            html += (
                f'<td style="border: 1px solid #dddddd; padding: 2px;'
                f'word-wrap: break-word; font-size: 10px; vertical-align: top;">{description}</td>\n'
            )

            image_filenames = row["images"]
            for img_filename in image_filenames:
                if img_filename:
                    if img_filename.startswith("https"):
                        img_src = img_filename
                    else:
                        img_src = os.path.join(os.path.basename(self.images_folder), img_filename)
                    if is_image_url(img_src):
                        img_tag = f'<img src="{img_src}" style="max-height: 250px;">'
                    else:
                        img_tag = f'<video controls style="max-height: 250px;"><source src="{img_src}" type="video/mp4"></video>'
                    html += f'<td style="border: 1px solid #dddddd; text-align: center; padding: 2px;">{img_tag}</td>\n'
                else:
                    html += '<td style="border: 1px solid #dddddd; padding: 2px;"></td>\n'

            num_empty_cells = self.max_images - len(image_filenames)
            for _ in range(num_empty_cells):
                html += '<td style="border: 1px solid #dddddd; padding: 2px;"></td>\n'
            html += "</tr>\n"

        html += "</table>"

        with open(self.filename, "w") as f:
            f.write(html)


if __name__ == "__main__":
    import random

    from PIL import Image

    def test_image_grid_visualizer():
        visualizer = ImageGridVisualizer("test_output.html")

        for i in range(5):
            description = f"Description for Image {i + 1}"
            imgs = []
            for seed in range(4):
                color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                img = Image.new("RGB", (100, 100), color)
                imgs.append(img)
            visualizer.add_row(description, imgs)

        visualizer.render()

        print("Test completed: 'test_output.html' file created with images and descriptions.")

    def test_image_grid_visualizer_links():
        visualizer = ImageGridVisualizer("test_output_links.html")

        for i in range(5):
            description = f"Description for Image {i + 1}"
            visualizer.add_row_urls(
                description, [f"gs://ml-team-evaluation/test_eval/t2i_generation/{i}.jpg" for i in range(1, 5)]
            )

        visualizer.render()

        print("Test completed: 'test_output.html' file created with images and descriptions.")

    def test_with_raw(descriptions):
        visualizer = ImageGridVisualizer("test_output_links_withprompts.html")

        for i, prompts in enumerate(descriptions):
            description = f"Prompt: {prompts}"
            visualizer.add_row_urls(
                description, [f"gs://ml-team-evaluation/test_eval/t2i_generation/{i}/{j}.jpg" for j in range(1, 5)]
            )

        visualizer.render()

        print("Test completed: 'test_output.html' file created with images and descriptions.")

    def test_video(descriptions):
        visualizer = ImageGridVisualizer("test_video_output_links_50_steps.html")

        for i, prompts in enumerate(descriptions):
            description = f"Prompt: {prompts}"
            visualizer.add_row_urls(
                description,
                [
                    f"gs://ml-team-evaluation/test_eval/t2i_generation/{i}/{j}_i2v_video_50_steps.mp4"
                    for j in range(1, 2)
                ],
            )

        visualizer.render()

        print("Test completed: 'test_output.html' file created with images and descriptions.")

    test_image_grid_visualizer()
    test_image_grid_visualizer_links()
    prompts = [
        "A solitary explorer walking across a vast golden desert at sunset, long shadows cast on rolling dunes under a fiery sky.",
        "A small herd of wild horses galloping across an open meadow of tall grass and wildflowers, their manes flowing under a gentle summer sun.",
        "A quiet corner of a library where an elderly woman reads a worn book, rays of morning light highlighting floating dust particles around her.",
        "Two tigers drinking at a clear forest stream in a lush green jungle, dappled sunlight passing through thick leaves overhead.",
        "A family sitting together at a wooden kitchen table in a cozy cottage, laughing softly as they share fresh bread and warm tea.",
        "A bustling early morning market square filled with vibrant fruit stalls, merchants arranging their wares, and customers chatting softly.",
        "A lone fisherman drifting on a calm lake at dawn, his small wooden boat gently rippling the reflection of distant mountain peaks.",
        "A flock of birds silhouetted against a warm evening sky, wings outstretched as they soar gracefully across the fading light.",
        "A winding garden path lined with rose bushes, leading to a quiet wooden bench where a child sits, lost in daydreams.",
        "A tranquil beach at twilight, gentle waves rolling onto the shore, a person strolling barefoot on soft sand under a lavender-tinted sky.",
    ]
    test_with_raw(prompts)
    test_video(prompts)

